"""Observation vs intervention in a host-parasite-parasitoid cascade (illustrative placeholder numbers).

Runs config.json with the library (scenarios, interventions, effects, observations, sensitivity), prints the
observed-vs-intervened table, and verifies every P(outcome | do(X=x)) independently with a plain numpy
truncated factorisation that does not use the library's inference code.

    python run_example.py                 # print the table and the check
    python run_example.py --out-dir out   # also write results.json / results.md / figures (needs matplotlib)
"""
import argparse
import os
import string

import numpy as np

from fxtensor_bayesnet import load_config, run

HERE = os.path.dirname(os.path.abspath(__file__))


def truncated_factorisation(cfg, do_var, do_label, target):
    """P(target | do(do_var=do_label)) = sum over all variables of prod_{v != do_var} P(v | pa(v)) * delta(do_var)."""
    V, P, K = cfg["variables"], cfg["priors"], cfg["kernels"]
    letter = {v: string.ascii_letters[i] for i, v in enumerate(V)}
    ops, subs = [], []
    for v, spec in V.items():
        labs = spec["labels"]
        if v == do_var:
            ops.append(np.array([1.0 if lab == do_label else 0.0 for lab in labs]))
            subs.append(letter[v])
        elif v in P:
            ops.append(np.array(P[v]["p"], float))
            subs.append(letter[v])
        else:
            pars = K[v]["parents"]
            arr = np.zeros([len(V[p]["labels"]) for p in pars] + [len(labs)])
            for idx in np.ndindex(*arr.shape[:-1]):
                arr[idx] = K[v]["cpt"]["|".join(V[p]["labels"][i] for p, i in zip(pars, idx))]
            ops.append(arr)
            subs.append("".join(letter[p] for p in pars) + letter[v])
    return np.einsum(",".join(subs) + "->" + letter[target], *ops)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    res = run(cfg)
    V = cfg["variables"]
    print("NOTE: all numbers are illustrative placeholders, not estimates from data or from any paper.\n")
    print(f"{'treatment':<12}{'level':<8}{'P(obs)':>9}{'P(do)':>9}")
    max_err = 0.0
    for key, e in res["因果効果"].items():
        y, ylab = e["結果"], e["結果のラベル"]
        for lev in V[e["処置"]]["labels"]:
            p_obs, p_do = e["P(結果|処置)(観測)"][lev], e["P(結果|do(処置))"][lev]
            ref = truncated_factorisation(cfg, e["処置"], lev, y)[V[y]["labels"].index(ylab)]
            max_err = max(max_err, abs(p_do - ref))
            print(f"{e['処置']:<12}{lev:<8}{p_obs:>9.4f}{p_do:>9.4f}")
        print(f"  -> observed difference {e['観測上の差'] * 100:+.1f}pt, ACE {e['ACE'] * 100:+.1f}pt, "
              f"gap (confounding) {e['交絡による差(観測上の差−ACE)'] * 100:+.1f}pt  [{key}]")
    print(f"\nmax |library do() - independent truncated factorisation| = {max_err:.2g} (outputs rounded to 4 d.p.)")
    assert max_err < 1e-4, max_err
    if a.out_dir:
        from fxtensor_bayesnet.cli import main as cli
        return cli(["--config", a.config, "--out-dir", a.out_dir, "--md", "--plot", "--lang", "en", "--json-keys", "en"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
