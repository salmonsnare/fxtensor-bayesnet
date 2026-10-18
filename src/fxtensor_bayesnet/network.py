"""ネットワーク：FXTensor による状態・マルコフ核の組み立てと、einsum による直接計算。"""
from __future__ import annotations

import string
from itertools import product

import numpy as np
from fxtensor_salmon import FXTensor

from .config import topo_order

MAX_WIRING = 1_000_000     # 配線核（密）の要素数上限。超えるときは同時分布の直接和 / einsum に切り替える


class Net:
    """設定から作るベイジアンネットワーク（各子変数の条件付き確率表 = マルコフ核）。"""

    def __init__(self, cfg):
        """設定から変数・ラベル・核を組み立てる。変数が 52 個を超えると ValueError。"""
        self.cfg = cfg
        self.order = topo_order(cfg)
        if len(self.order) > len(string.ascii_letters):
            raise ValueError(f"変数が多すぎます（最大 {len(string.ascii_letters)}）")
        self.labels = {v: cfg["variables"][v]["labels"] for v in self.order}
        self.roots = [v for v in self.order if v in cfg["priors"]]
        self.parents = {v: cfg["kernels"][v]["parents"] for v in self.order if v in cfg["kernels"]}
        self.kernels = {}      # v → FXTensor  (親1⊗…⊗親k → v)
        self.cpt = {}          # v → ndarray  (親…, v)
        for v, pars in self.parents.items():
            rows = cfg["kernels"][v]["cpt"]
            arr = np.zeros([len(self.labels[p]) for p in pars] + [len(self.labels[v])])
            for idx in product(*[range(len(self.labels[p])) for p in pars]):
                arr[idx] = rows["|".join(self.labels[p][i] for p, i in zip(pars, idx))]
            k = FXTensor([[self.labels[p] for p in pars], [self.labels[v]]], data=arr)
            assert k.is_markov(), v
            self.kernels[v], self.cpt[v] = k, arr
        self.letter = {v: string.ascii_letters[i] for i, v in enumerate(self.order)}

    # --- 祖先集合 ---
    def ancestors(self, names, cut=()):
        """names の祖先集合（自身を含む）。cut の変数は親への辺を切ったものとして扱う（do のグラフ手術）。"""
        out, stack = set(), list(names)
        while stack:
            v = stack.pop()
            if v in out:
                continue
            out.add(v)
            if v not in cut:
                stack += self.parents.get(v, [])
        return [v for v in self.order if v in out]

    def joint_size(self):
        """全変数の同時分布の要素数（各変数のラベル数の積）。"""
        return int(np.prod([len(self.labels[v]) for v in self.order]))

    # --- einsum による直接計算（検算・メモリ節約経路） ---
    def contract(self, priors, targets, evidence=None, do=None, cpt=None, free_roots=False):
        """変数消去：祖先集合の因子だけを einsum で縮約し、targets の（正規化しない）同時分布を返す。

        evidence: {変数: 尤度ベクトル}（ハード証拠は one-hot）。結果は P(targets, evidence) になる。
        do: {変数: 分布}。その変数の核を分布で置き換える（グラフ手術、切断因子分解）。
        cpt: {変数: ndarray}。核（または根の事前分布）を差し替える（感度分析用）。
        free_roots=True なら 根→targets の核（根の事前分布を掛けない）。
        同時分布全体は作らない（np.einsum の optimize で小さい中間テンソルだけを作る）。
        """
        evidence, do, cpt = evidence or {}, do or {}, cpt or {}
        anc = self.ancestors(list(targets) + list(evidence), cut=set(do))
        ops, subs = [], []
        for v in anc:
            if v in do:
                ops.append(np.asarray(do[v], float))
                subs.append(self.letter[v])
            elif v in self.parents:
                ops.append(cpt.get(v, self.cpt[v]))
                subs.append("".join(self.letter[p] for p in self.parents[v]) + self.letter[v])
            elif not free_roots:
                ops.append(np.asarray(cpt.get(v, priors[v]), float))
                subs.append(self.letter[v])
        for v, lik in evidence.items():
            ops.append(np.asarray(lik, float))
            subs.append(self.letter[v])
        if free_roots:   # 祖先に入らない根も核の入力にする（その根には依存しない＝定数）
            for r in self.roots:
                ops.append(np.ones(len(self.labels[r])))
                subs.append(self.letter[r])
        out = "".join(self.letter[r] for r in self.roots) if free_roots else ""
        out += "".join(self.letter[t] for t in targets)
        if not ops:
            raise ValueError("empty query")
        return np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)

    def einsum_query(self, priors, targets, free_roots=False):
        """祖先集合だけを使い、targets の同時分布（free_roots=True なら 根→targets の核）を einsum で直接計算。"""
        return self.contract(priors, targets, free_roots=free_roots)

    def brute_force_joint(self, priors):
        """全変数の同時分布を einsum 一発で（検算用）。軸の順は self.order。"""
        ops, subs = [], []
        for v in self.order:
            if v in self.parents:
                ops.append(self.cpt[v])
                subs.append("".join(self.letter[p] for p in self.parents[v]) + self.letter[v])
            else:
                ops.append(np.asarray(priors[v], float))
                subs.append(self.letter[v])
        return np.einsum(",".join(subs) + "->" + "".join(self.letter[v] for v in self.order), *ops, optimize=True)

    # --- ライブラリによる組み立て ---
    def prior_state(self, priors):
        """根の事前分布を FXTensor の状態（tensor_product）にする。priors は {根: 確率のリスト}。"""
        st = None
        for r in self.roots:
            s = FXTensor([[], [self.labels[r]]], data=np.asarray(priors[r], float))
            st = s if st is None else st.tensor_product(s)
        return st

    def build_joint(self, priors):
        """根の状態（tensor_product）に、トポロジカル順に子を graph_extend で足して同時状態を作る。

        軸の順は「根（self.roots の順）→ 子（トポロジカル順）」。self.order とは一致しないことがある。"""
        st = self.prior_state(priors)
        names = list(self.roots)
        for v in self.order:
            if v in self.parents:
                st = graph_extend(st, self.kernels[v], [names.index(p) for p in self.parents[v]])
                names.append(v)
        return st, names


def wiring_size(in_labels, out_idx):
    """wiring(in_labels, out_idx) が作る密テンソルの要素数。"""
    dims = [len(lab) for lab in in_labels]
    return int(np.prod(dims)) * int(np.prod([dims[i] for i in out_idx] or [1]))


def wiring(in_labels, out_idx):
    """決定的な配線核 X1⊗…⊗Xn → X_{out_idx}（copy / swap / discard の一般形・ラベル付き）。"""
    dims = [len(lab) for lab in in_labels]
    size = int(np.prod(dims)) * int(np.prod([dims[i] for i in out_idx] or [1]))
    if size > MAX_WIRING:
        raise MemoryError(f"配線核が大きすぎます（{size} 要素）")
    data = np.zeros(dims + [dims[i] for i in out_idx])
    idx = np.indices(dims).reshape(len(dims), -1)
    data[tuple(idx) + tuple(idx[i] for i in out_idx)] = 1.0
    return FXTensor([list(in_labels), [in_labels[i] for i in out_idx]], data=data)


def graph_extend(st, k, parent_axes):
    """状態 I→X に核 k:(親)→Y を足して I→X⊗Y（= st ; copy ; (id ⊗ (選択;k))）。密な恒等を作らない。"""
    n = st.data.ndim
    xs = string.ascii_letters[:n]
    y = string.ascii_letters[n]
    data = np.einsum(f"{xs},{''.join(xs[i] for i in parent_axes)}{y}->{xs}{y}", st.data, k.data)
    return FXTensor([[], st.labels[1] + k.labels[1]], data=data)


def self_test_graph_extend():
    """graph_extend と、ライブラリだけで書いた p ; copy ; (id ⊗ f) の一致を確認。"""
    A, B = ["a0", "a1", "a2"], ["b0", "b1"]
    rng = np.random.default_rng(0)
    p = FXTensor([[], [A, B]], data=(lambda x: x / x.sum())(rng.random((3, 2))))
    f = FXTensor([[A], [B]], data=(lambda x: x / x.sum(1, keepdims=True))(rng.random((3, 2))))
    lib = p.composition(wiring([A, B], [0, 1, 0])).composition(FXTensor.identity_tensor([A, B]).tensor_product(f))
    assert np.allclose(lib.data, graph_extend(p, f, [0]).data)
