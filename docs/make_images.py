"""Generate the README figures into docs/images/ (Japanese) and docs/images/en/ (English).

    pip install -e '.[plot]'
    python docs/make_images.py
"""
import copy
import os

from fxtensor_bayesnet import load_config, run
from fxtensor_bayesnet.plot import plot_diff, plot_effects, plot_marginals, plot_network, plot_sensitivity, plot_updates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (lang, config file name, output dir, diff-example priors)
VARIANTS = [
    ("ja", "config.json", os.path.join(ROOT, "docs", "images")),
    ("en", "config.en.json", os.path.join(ROOT, "docs", "images", "en")),
]


def main():
    for lang, fname, out in VARIANTS:
        for name in ("weather", "maintenance"):
            cfg = load_config(os.path.join(ROOT, "examples", name, fname))
            res = run(cfg)
            plot_network(cfg, os.path.join(out, f"{name}_network.png"), lang=lang)
            plot_marginals(res, cfg, os.path.join(out, f"{name}_marginals.png"), lang=lang)
            plot_updates(res, cfg, os.path.join(out, f"{name}_updates.png"), lang=lang)

        # Example for --diff --plot: the next day, the pressure and humidity priors are revised
        cfg = load_config(os.path.join(ROOT, "examples", "weather", fname))
        new = copy.deepcopy(cfg)
        new["as_of"] = "2026-10-11"
        new["priors"]["pressure"]["p"] = [0.25, 0.3, 0.45]
        new["priors"]["humidity"]["p"] = [0.2, 0.45, 0.35]
        plot_diff(run(cfg), run(new), new, os.path.join(out, "weather_diff.png"),
                  old_label=cfg["as_of"], new_label=new["as_of"], lang=lang)
        print(out, sorted(f for f in os.listdir(out) if f.endswith(".png")))

    ecology_images()


def ecology_images():
    """English-only example: observed vs intervened (effects) and sensitivity (tornado)."""
    out = os.path.join(ROOT, "docs", "images", "en")
    cfg = load_config(os.path.join(ROOT, "examples", "ecology_host_parasitoid", "config.json"))
    res = run(cfg)
    plot_effects(res, cfg, os.path.join(out, "ecology_effects.png"), lang="en")
    plot_sensitivity(res, cfg, os.path.join(out, "ecology_sensitivity.png"), lang="en")


if __name__ == "__main__":
    import sys
    ecology_images() if "--ecology-only" in sys.argv else main()
