#!/usr/bin/env python3
"""Compare cross-platform determinism dumps from every CI leg.

usage: compare-determinism.py <dir> [--expect N]

<dir> holds one sub-directory per CI leg, named <os>-<compiler>-<Config>
(e.g. linux-gcc14-Release), each containing the determinism-*.txt files that
scripts/run-tests.{sh,ps1} produced via tests/CrossPlatformDeterminismTest.cpp.

Rules (see docs/ci.md, "Determinism"):
  * every dump must list the same scenes;
  * class "trigfree" scenes must be bit-identical across ALL legs (every OS,
    compiler, configuration and seed);
  * class "trig" scenes must be bit-identical within each OS (one libm), and are
    reported -- not failed -- when they differ between OSes.
Exit status 1 on any rule violation, 0 otherwise. Writes a Markdown report to
$GITHUB_STEP_SUMMARY when that variable is set.
"""

import argparse
import os
import sys
from collections import defaultdict


def parse(path):
    scenes = {}
    states = defaultdict(list)
    with open(path, encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "scene" and len(parts) == 4:
                scenes[parts[1]] = (parts[2], parts[3])
            elif parts[0] == "state" and len(parts) == 6:
                states[parts[1]].append(tuple(float.fromhex(v) for v in parts[3:]))
            else:
                raise ValueError(f"{path}: malformed line: {line.rstrip()}")
    return scenes, states


def max_abs_diff(a, b):
    worst = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            worst = max(worst, abs(va - vb))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--expect", type=int, default=0, help="minimum number of legs")
    args = ap.parse_args()

    dumps = {}  # "<leg>/<file>" -> (os, scenes, states)
    legs = set()
    for leg in sorted(os.listdir(args.dir)):
        leg_dir = os.path.join(args.dir, leg)
        if not os.path.isdir(leg_dir):
            continue
        for name in sorted(os.listdir(leg_dir)):
            if name.startswith("determinism-") and name.endswith(".txt"):
                legs.add(leg)
                scenes, states = parse(os.path.join(leg_dir, name))
                dumps[f"{leg}/{name}"] = (leg.split("-", 1)[0], scenes, states)

    errors = []
    report = ["## Cross-platform determinism", ""]
    if len(legs) < args.expect:
        errors.append(f"expected >= {args.expect} legs with dumps, found {len(legs)}: {sorted(legs)}")
    if not dumps:
        errors.append("no determinism dumps found")

    scene_sets = {frozenset(s) for _, s, _ in dumps.values()}
    if len(scene_sets) > 1:
        errors.append(f"dumps disagree on the scene list: {[sorted(s) for s in scene_sets]}")
    all_scenes = sorted(set().union(*scene_sets)) if scene_sets else []

    report.append("| scene | class | result |")
    report.append("|---|---|---|")
    for scene in all_scenes:
        classes = {s[scene][0] for _, s, _ in dumps.values() if scene in s}
        cls = classes.pop() if len(classes) == 1 else "?"
        if cls == "?":
            errors.append(f"{scene}: inconsistent class across dumps")
            continue
        by_hash = defaultdict(list)
        by_os_hash = defaultdict(lambda: defaultdict(list))
        for key, (osname, s, _) in dumps.items():
            if scene in s:
                by_hash[s[scene][1]].append(key)
                by_os_hash[osname][s[scene][1]].append(key)

        if len(by_hash) == 1:
            report.append(f"| {scene} | {cls} | identical on all {len(dumps)} dumps (`{next(iter(by_hash))}`) |")
            continue

        # Within-OS identity is required for every class.
        for osname, hashes in sorted(by_os_hash.items()):
            if len(hashes) > 1:
                errors.append(f"{scene} ({cls}): differs WITHIN {osname}: "
                              + "; ".join(f"{h}: {', '.join(k)}" for h, k in hashes.items()))

        # Magnitude of the cross-OS difference, from the final-state hex floats.
        rep = {osname: next(iter(next(iter(h.values())))) for osname, h in by_os_hash.items()}
        names = sorted(rep)
        mags = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = dumps[rep[names[i]]][2][scene]
                b = dumps[rep[names[j]]][2][scene]
                mags.append(f"{names[i]} vs {names[j]}: max |d| = {max_abs_diff(a, b):.3g} m/rad")
        detail = "; ".join(f"{o}=`{next(iter(h))}`" for o, h in sorted(by_os_hash.items()))
        if cls == "trigfree":
            errors.append(f"{scene} (trigfree) differs across legs: {detail} ({'; '.join(mags)})")
            report.append(f"| {scene} | {cls} | **FAIL** differs: {detail} |")
        else:
            report.append(f"| {scene} | {cls} | per-OS identical, differs across libms (expected): "
                          f"{detail}; {'; '.join(mags)} |")

    report.append("")
    report.append(f"Legs: {', '.join(sorted(legs))}")
    if errors:
        report += ["", "### Violations", ""] + [f"- {e}" for e in errors]

    text = "\n".join(report) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
