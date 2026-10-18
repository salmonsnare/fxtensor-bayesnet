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
        "obs_header_soft": "### {obs} を観測（尤度の期待値 {p:.4g}）",
        "impossible": "確率 0 の観測のため条件付けできません（この観測の更新は省略）",
        "ranking": "動きの大きさ順(TV)",
        "delta": "差",
        "checks": "検算",
        "not_checked": "未検算（比べていないので最大差は null）: {keys}",
        "skipped_obs": "検算を省略した観測（{kind}）: {obs}",
        "network_title": "ネットワーク構造",
        "kind_root": "根（事前分布）",
        "kind_middle": "中間変数",
        "kind_output": "出力変数",
        "marginals_title": "周辺分布のシナリオ比較（基準日 {as_of}）",
        "prior": "事前",
        "posterior": "事後",
        "obs_annot": "観測: {obs}（観測前の確率 {p:.1f}%）",
        "obs_annot_soft": "観測: {obs}（尤度の期待値 {p:.4g}）",
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
        "zero_level_md": "処置の水準 {levs} は観測上の確率が 0 のため、観測の値は未定義（—）。0 ではありません",
        "zero_level_bar": "未定義\n(P=0)",
        "zero_level_note": "— : 未定義（観測確率 0 の水準）",
        "strat_do_header": "#### 層別 ACE（{z} を介入で固定：given_do）",
        "strat_obs_header": "#### 条件付き ACE（{z} を観測で条件付け：given）",
        "stratum": "層",
        "strat_do_col": "P({y} ∣ do({x}={lev}), do(層))",
        "strat_obs_col": "P({y} ∣ do({x}={lev}), 層)",
        "strat_diff": "層間の ACE の差（{last} − {first}）{d}",
        "strat_undefined": "— は、その層が do({x}) の下で確率 0 のため未定義",
        "strat_do_title": "層別 ACE：do({z}) で固定",
        "strat_obs_title": "条件付き ACE：{z} で条件付け",
        "overall_ace": "全体の ACE {ace}pt",
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
        "obs_header_soft": "### Observed {obs} (expected likelihood {p:.4g})",
        "impossible": "This observation has probability 0 and cannot be conditioned on (update skipped)",
        "ranking": "Ranking by movement (TV)",
        "delta": "change",
        "checks": "Checks",
        "not_checked": "Not checked (no comparison was made, so the max error is null): {keys}",
        "skipped_obs": "Observations not cross-checked ({kind}): {obs}",
        "network_title": "Network structure",
        "kind_root": "Root (prior)",
        "kind_middle": "Intermediate",
        "kind_output": "Output",
        "marginals_title": "Marginals by scenario (as of {as_of})",
        "prior": "Prior",
        "posterior": "Posterior",
        "obs_annot": "Observed: {obs} (prior probability {p:.1f}%)",
        "obs_annot_soft": "Observed: {obs} (expected likelihood {p:.4g})",
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
        "zero_level_md": "Treatment level(s) {levs} have probability 0, so the observed value is undefined (—), not 0",
        "zero_level_bar": "undefined\n(P=0)",
        "zero_level_note": "— : undefined (level with probability 0)",
        "strat_do_header": "#### Stratified ACE ({z} fixed by intervention: given_do)",
        "strat_obs_header": "#### Conditional ACE ({z} observed: given)",
        "stratum": "Stratum",
        "strat_do_col": "P({y} ∣ do({x}={lev}), do(stratum))",
        "strat_obs_col": "P({y} ∣ do({x}={lev}), stratum)",
        "strat_diff": "Difference in ACE between strata ({last} − {first}) {d}",
        "strat_undefined": "— : undefined, the stratum has probability 0 under do({x})",
        "strat_do_title": "Stratified ACE: do({z})",
        "strat_obs_title": "Conditional ACE: given {z}",
        "overall_ace": "overall ACE {ace}pt",
        "sens_title": "Sensitivity (tornado): P({y})",
        "sens_low": "parameter lowered",
        "sens_high": "parameter raised",
        "na": "—",
    },
}


def t(lang, key, **kw):
    """表示文言 ``TEXT[lang][key]`` を返す。kw があれば format する。未対応の lang は ValueError。"""
    if lang not in TEXT:
        raise ValueError(f"unsupported language: {lang} (choose from {', '.join(LANGS)})")
    s = TEXT[lang][key]
    return s.format(**kw) if kw else s


def scenario_name(lang, name):
    """結果内部のシナリオ名「基本」を表示言語に合わせる。"""
    return t(lang, "base") if name == "基本" else name


def obs_text(lang, kind, key, r):
    """観測の見出し（kind="obs_header" | "obs_annot"）。ハード証拠だけなら観測前の確率（%）、ソフト証拠を含むと尤度の期待値。"""
    if "尤度の期待値" in r:
        return t(lang, kind + "_soft", obs=obs_name(key), p=r["尤度の期待値"])
    return t(lang, kind, obs=obs_name(key), p=r["観測前の確率"] * 100)


def obs_name(key):
    """結果キー「var=label を観測」から「var=label」を取り出す。"""
    return key.replace(" を観測", "")
