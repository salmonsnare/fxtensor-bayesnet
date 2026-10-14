"""results.json のキーの英語版（--json-keys en / to_english_keys）。

内部の結果辞書と履歴のスナップショットは従来どおり日本語キー（正準形）。英語キーは出力時だけの変換で、
変数名・ラベル名・シナリオ名など利用者が付けた名前は変えない（位置で判別するので、ラベルが「事後」等でも安全）。
"""
from __future__ import annotations

_MOVE = {"差(pt)": "delta_pt", "全変動距離TV": "tv_distance", "KL(事後||事前)[nat]": "kl_posterior_prior_nat"}
_ANY = (None, None)       # 名前はそのまま・中身もそのまま


def _obs_key(k):
    return k[:-len(" を観測")] if k.endswith(" を観測") else k


_SCENARIO = {"上書き": "overrides", "周辺": "marginals"}
_UPDATE = {
    "証拠": ("evidence", {"*": (None, {"尤度": "likelihood"})}),
    "観測前の確率": "evidence_probability",
    "尤度の期待値": "expected_likelihood",
    "不可能な観測": "impossible",
    "事後": "posterior",
    "注目変数の事前": "focus_prior",
    "注目変数の動き": ("focus_movement", {"*": (None, _MOVE)}),
    "動きの大きさ順(TV)": "ranking_by_tv",
}
_EFFECT = {
    "処置": "treatment", "結果": "outcome", "結果のラベル": "outcome_label", "処置群": "treated", "対照群": "control",
    "P(結果|do(処置))": "p_outcome_do", "P(結果|処置)(観測)": "p_outcome_observed", "ACE": "ace",
    "観測上の差": "observed_difference", "交絡による差(観測上の差−ACE)": "confounding_gap",
    "観測確率0の水準": ("zero_probability_levels", None),
    "名前": "name",
    "層別ACE(given_do)": ("stratified_ace_given_do", {
        "条件": ("given", None), "層": ("strata", {"*": (None, {
            "条件の値": ("given_values", None), "P(結果|do(処置),do(層))": "p_outcome_do_given_do", "ACE": "ace"})}),
        "層間のACEの差(最後−最初)": "ace_difference_last_minus_first"}),
    "条件付きACE(given)": ("conditional_ace_given", {
        "条件": ("given", None), "層": ("strata", {"*": (None, {
            "条件の値": ("given_values", None), "P(層|do(処置))": "p_stratum_do",
            "P(結果|do(処置),層)": "p_outcome_do_given", "ACE": "ace"})}),
        "層間のACEの差(最後−最初)": "ace_difference_last_minus_first"}),
}
_SENS_ROW = {"パラメータ": "parameter", "基準値": "base_value", "下げた値": "low_value", "上げた値": "high_value",
             "P(下げたとき)": "p_low", "P(上げたとき)": "p_high", "幅(pt)": "swing_pt"}
_SENS = {"目的": "target", "ラベル": "label", "基準のP": "baseline", "変化量delta": "delta",
         "パラメータ数": "n_parameters", "上位": ("top", _SENS_ROW)}
_CHECKS = {
    "計算モード": "mode", "同時分布の要素数": "joint_size",
    "einsum(全同時)との最大差": "max_err_joint_vs_einsum",
    "einsum(祖先集合・周辺)との最大差": "max_err_marginal_vs_einsum",
    "einsum(条件付け)との最大差": "max_err_conditioning_vs_einsum",
    "bayesian_inversionとconditionalizationの最大差": "max_err_inversion_vs_conditionalization",
    "介入:切断因子分解との最大差": "max_err_intervention_vs_truncated_factorization",
    "bayesian_inversionの検算を省略した観測": ("inversion_check_skipped", None),
    "条件付けの検算を省略した観測": ("conditioning_check_skipped", None),
    "備考": "remark",
}
SCHEMA = {
    "基準日": "as_of", "ホライズン": "horizon", "注意": "note", "設定ハッシュ": "config_hash",
    "事前分布(基本)": "priors_base",
    "シナリオ": ("scenarios", {"基本": ("base", _SCENARIO), "*": (None, _SCENARIO)}),
    "ベイズ更新(基本)": ("updates_base", {"*": (_obs_key, _UPDATE)}),
    "介入": ("interventions", {"*": (None, {"介入内容": "do", "周辺": "marginals"})}),
    "因果効果": ("effects", {"*": (None, _EFFECT)}),
    "感度分析": ("sensitivity", {"*": (None, _SENS)}),
    "検算": ("checks", _CHECKS),
}

#: 日本語キー → 英語キー（固定のキーだけ。README の対応表と同じ）
KEY_MAP = {}


def _collect(schema):
    for k, spec in schema.items():
        if k == "*":
            spec[1] and _collect(spec[1])
            continue
        name, sub = (spec, None) if isinstance(spec, str) else spec
        if isinstance(name, str):
            KEY_MAP[k] = name
        if sub:
            _collect(sub)


_collect(SCHEMA)


def _tr(obj, schema):
    if schema is None:
        return obj
    if isinstance(obj, list):
        return [_tr(x, schema) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        spec = schema.get(k, schema.get("*"))
        if spec is None:
            out[k] = v
            continue
        name, sub = (spec, None) if isinstance(spec, str) else spec
        name = k if name is None else name(k) if callable(name) else name
        out[name] = _tr(v, sub)
    return out


def to_english_keys(res):
    """run() の結果（日本語キー）を英語キーに変換した新しい辞書を返す。利用者が付けた名前と値は変えない。
    観測のキー「var=label を観測」は「var=label」になる。"""
    return _tr(res, SCHEMA)
