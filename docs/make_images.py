"""README 用の図を docs/images/ に生成する。

    pip install -e '.[plot]'
    python docs/make_images.py
"""
import copy
import os

from fxtensor_bayesnet import load_config, run
from fxtensor_bayesnet.plot import plot_diff, plot_marginals, plot_network, plot_updates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")


def main():
    for name in ("weather", "maintenance"):
        cfg = load_config(os.path.join(ROOT, "examples", name, "config.json"))
        res = run(cfg)
        plot_network(cfg, os.path.join(OUT, f"{name}_network.png"))
        plot_marginals(res, cfg, os.path.join(OUT, f"{name}_marginals.png"))
        plot_updates(res, cfg, os.path.join(OUT, f"{name}_updates.png"))

    # --diff の図の例：天気の翌日、気圧配置と湿度の事前分布を見直した場合
    cfg = load_config(os.path.join(ROOT, "examples", "weather", "config.json"))
    new = copy.deepcopy(cfg)
    new["as_of"] = "2026-10-11"
    new["priors"]["pressure"]["p"] = [0.25, 0.3, 0.45]
    new["priors"]["humidity"]["p"] = [0.2, 0.45, 0.35]
    plot_diff(run(cfg), run(new), new, os.path.join(OUT, "weather_diff.png"),
              old_label=cfg["as_of"], new_label=new["as_of"])
    print("\n".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
