"""表示用の文言（Markdown と図）。results.json のキーは --lang によらない（英語キーは --json-keys en）。"""
from __future__ import annotations

LANGS = ("ja", "en")

TEXT = {
    "ja": {
        "result_title": "# 結果（基準日 {as_of}{horizon}）",
        "horizon_sep": "、",
        "scenario": "シナリオ",
        "base": "基本",
        "bayes_header": "## ベイズ更新（基本）",
        "obs_header": "### {obs} を観測（観測前の確率 {p:.1f}%）",
        "ranking": "動きの大きさ順(TV)",
        "delta": "差",
        "checks": "検算",
        "network_title": "ネットワーク構造",
        "kind_root": "根（事前分布）",
        "kind_middle": "中間変数",
        "kind_output": "出力変数",
        "marginals_title": "周辺分布のシナリオ比較（基準日 {as_of}）",
        "prior": "事前",
        "posterior": "事後",
        "obs_annot": "観測: {obs}（観測前の確率 {p:.1f}%）",
        "updates_title": "ベイズ更新：注目変数の事前・事後",
        "change_pt": "変化（pt）",
        "no_change": "変化なし",
        "diff_title": "周辺分布の変化：{old} → {new}",
        "previous": "前回",
        "current": "今回",
        "interventions_header": "## 介入（do：グラフ手術）",
        "intervention": "介入",
        "effects_header": "## 因果効果（観測による条件付けと介入の比較）",
        "level": "水準",
        "observed_col": "P({y} ∣ {x}=水準)（観測）",
        "do_col": "P({y} ∣ do({x}=水準))（介入）",
        "effect_line": "観測上の差（{t}−{c}）{od} / 平均因果効果 ACE {ace} / 差（交絡による分）{gap}",
        "sens_header": "## 感度分析：P({y}) = {p:.1f}%（各パラメータ ±{d}、{n} 個中の上位 {k} 個）",
        "sens_cols": "| パラメータ | 基準値 | 下げたとき | 上げたとき | 幅（pt） |",
        "effects_title": "観測（条件付け）と介入（do）の比較",
        "observed": "観測 P(Y | X=x)",
        "intervened": "介入 P(Y | do(X=x))",
        "effect_annot": "観測差 {od}pt / ACE {ace}pt",
        "sens_title": "感度分析（トルネード図）：P({y})",
        "sens_low": "パラメータを下げたとき",
        "sens_high": "パラメータを上げたとき",
        "na": "—",
    },
    "en": {
        "result_title": "# Results (as of {as_of}{horizon})",
        "horizon_sep": ", ",
        "scenario": "Scenario",
        "base": "Base",
        "bayes_header": "## Bayesian updates (base)",
        "obs_header": "### Observed {obs} (prior probability {p:.1f}%)",
        "ranking": "Ranking by movement (TV)",
        "delta": "change",
        "checks": "Checks",
        "network_title": "Network structure",
        "kind_root": "Root (prior)",
        "kind_middle": "Intermediate",
        "kind_output": "Output",
        "marginals_title": "Marginals by scenario (as of {as_of})",
        "prior": "Prior",
        "posterior": "Posterior",
        "obs_annot": "Observed: {obs} (prior probability {p:.1f}%)",
        "updates_title": "Bayesian update: prior vs posterior of focus variables",
        "change_pt": "Change (pt)",
        "no_change": "No change",
        "diff_title": "Change in marginals: {old} → {new}",
        "previous": "previous",
        "current": "current",
        "interventions_header": "## Interventions (do: graph surgery)",
        "intervention": "Intervention",
        "effects_header": "## Causal effects (conditioning vs intervention)",
        "level": "Level",
        "observed_col": "P({y} ∣ {x}=level) (observed)",
        "do_col": "P({y} ∣ do({x}=level)) (intervened)",
        "effect_line": "Observed difference ({t} − {c}) {od} / average causal effect ACE {ace} / gap (confounding) {gap}",
        "sens_header": "## Sensitivity: P({y}) = {p:.1f}% (each parameter ±{d}, top {k} of {n})",
        "sens_cols": "| Parameter | Base value | Lowered | Raised | Swing (pt) |",
        "effects_title": "Observation (conditioning) vs intervention (do)",
        "observed": "Observed P(Y | X=x)",
        "intervened": "Intervened P(Y | do(X=x))",
        "effect_annot": "observed diff {od}pt / ACE {ace}pt",
        "sens_title": "Sensitivity (tornado): P({y})",
        "sens_low": "parameter lowered",
        "sens_high": "parameter raised",
        "na": "—",
    },
}


def t(lang, key, **kw):
    if lang not in TEXT:
        raise ValueError(f"unsupported language: {lang} (choose from {', '.join(LANGS)})")
    s = TEXT[lang][key]
    return s.format(**kw) if kw else s


def scenario_name(lang, name):
    """結果内部のシナリオ名「基本」を表示言語に合わせる。"""
    return t(lang, "base") if name == "基本" else name


def obs_name(key):
    """結果キー「var=label を観測」から「var=label」を取り出す。"""
    return key.replace(" を観測", "")
