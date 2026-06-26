"""Image-level, multi-label stratified 60/20/20 split for the ACRE YOLO data.

Why multi-label stratification: most images contain several of the 6 species at
once (and lambsquarter is in almost every image), so a plain random split can
starve a split of the rare species (mustard/ryegrass/matricaria/maize). We use
the greedy iterative stratification of Sechidis et al. (2011): process images
rarest-label-first and send each to the split that most needs its rarest label.

Outputs under paths.out_root:
  splits.json                          {train:[stems], val:[...], test:[...]}
  split_report.json                    per-split image + per-class instance counts
  fine/  coarse/  {images,labels}/{train,val,test}/  (symlink trees)
  acre_fine.yaml  acre_coarse.yaml     Ultralytics dataset configs
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import load_config  # noqa: E402


def iterative_stratify(image_labels: dict, ratios: dict, seed: int) -> dict:
    """image_labels: stem -> set(class_ids).  Returns stem -> split name."""
    splits = list(ratios)
    rng = random.Random(seed)

    label_total = Counter()
    for labs in image_labels.values():
        for l in labs:
            label_total[l] += 1
    n_total = len(image_labels)

    # desired remaining counts per split, per label, and overall size
    desired = {s: {l: label_total[l] * ratios[s] for l in label_total} for s in splits}
    desired_size = {s: n_total * ratios[s] for s in splits}
    assigned = {s: 0 for s in splits}
    assignment = {}

    # rarest-label-first; random tie-break for stability
    def rarity(stem):
        labs = image_labels[stem]
        return (min((label_total[l] for l in labs), default=10 ** 9), rng.random())

    for stem in sorted(image_labels, key=rarity):
        labs = image_labels[stem]
        if not labs:  # background-only / OOV-only image -> fill emptiest split
            s = max(splits, key=lambda s: (desired_size[s] - assigned[s], rng.random()))
        else:
            rare = min(labs, key=lambda l: label_total[l])
            s = max(splits, key=lambda s: (desired[s][rare], desired_size[s] - assigned[s], rng.random()))
            for l in labs:
                desired[s][l] -= 1
        assignment[stem] = s
        assigned[s] += 1
        desired_size[s] -= 1
    return assignment


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def build_tree(out_root: Path, label_set: str, assignment: dict):
    """label_set in {'fine','coarse'} -> Ultralytics images/labels/<split> tree."""
    lbl_src_dir = out_root / f"labels_{label_set}"
    for stem, split in assignment.items():
        _link(out_root / "images" / f"{stem}.jpg",
              out_root / label_set / "images" / split / f"{stem}.jpg")
        _link(lbl_src_dir / f"{stem}.txt",
              out_root / label_set / "labels" / split / f"{stem}.txt")


def write_yaml(out_root: Path, label_set: str, names: list):
    p = out_root / f"acre_{label_set}.yaml"
    lines = [
        f"# Ultralytics dataset config (auto-generated) - {label_set} labels",
        f"path: {out_root / label_set}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        f"names: {names}",
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


def main(cfg_path="configs/data.yaml"):
    cfg = load_config(cfg_path)
    out_root = Path(cfg["paths"]["out_root"])
    fine_names = cfg["fine_names"]
    coarse_names = cfg["coarse_names"]

    master = json.load(open(out_root / "annotations_master.json"))

    # per-image fine-label presence (stratify on the PRIMARY 6-class task)
    image_labels = defaultdict(set)
    all_stems = set()
    for box in master:
        all_stems.add(box["stem"])
        if box["fine_id"] is not None:
            image_labels[box["stem"]].add(box["fine_id"])
    # include images with zero fine boxes (empty or OOV-only) so they get split too
    for stem in all_stems:
        image_labels.setdefault(stem, set())
    # also the 2 truly-empty images may be absent from master entirely
    for txt in (out_root / "labels_fine").glob("*.txt"):
        image_labels.setdefault(txt.stem, set())

    assignment = iterative_stratify(image_labels, cfg["split"]["ratios"], cfg["split"]["seed"])

    # ---- splits.json ----
    splits = defaultdict(list)
    for stem, s in assignment.items():
        splits[s].append(stem)
    for s in splits:
        splits[s].sort()
    (out_root / "splits.json").write_text(json.dumps(splits, indent=2))

    # ---- per-split, per-class report (instances + images containing class) ----
    inst = {s: Counter() for s in splits}            # fine instances
    img_with = {s: Counter() for s in splits}        # images containing class
    coarse_inst = {s: Counter() for s in splits}
    for box in master:
        s = assignment[box["stem"]]
        if box["fine_id"] is not None:
            inst[s][box["fine_id"]] += 1
        coarse_inst[s][box["coarse_id"]] += 1
    for stem, labs in image_labels.items():
        s = assignment[stem]
        for l in labs:
            img_with[s][l] += 1

    report = {"images_per_split": {s: len(v) for s, v in splits.items()},
              "fine_instances_per_split": {}, "fine_images_with_class_per_split": {},
              "coarse_instances_per_split": {}}
    for i, nm in enumerate(fine_names):
        report["fine_instances_per_split"][nm] = {s: inst[s][i] for s in splits}
        report["fine_images_with_class_per_split"][nm] = {s: img_with[s][i] for s in splits}
    for i, nm in enumerate(coarse_names):
        report["coarse_instances_per_split"][nm] = {s: coarse_inst[s][i] for s in splits}
    (out_root / "split_report.json").write_text(json.dumps(report, indent=2))

    # ---- build Ultralytics trees + yamls (fine = primary, coarse = secondary) ----
    for label_set, names in [("fine", fine_names), ("coarse", coarse_names)]:
        build_tree(out_root, label_set, assignment)
        write_yaml(out_root, label_set, names)

    # ---- console ----
    print("images per split:", report["images_per_split"])
    print("\nFINE instances per split (and % of class in each split):")
    tot = Counter()
    for box in master:
        if box["fine_id"] is not None:
            tot[box["fine_id"]] += 1
    hdr = f"{'class':14s}" + "".join(f"{s:>10s}" for s in splits) + f"{'total':>9s}"
    print(hdr)
    for i, nm in enumerate(fine_names):
        row = f"{nm:14s}" + "".join(f"{inst[s][i]:>10d}" for s in splits) + f"{tot[i]:>9d}"
        print(row)
    print("\nimages containing each class (per split):")
    for i, nm in enumerate(fine_names):
        print(f"  {nm:14s}" + "".join(f"{s}={img_with[s][i]:<5d} " for s in splits))
    print(f"\nwrote splits.json, split_report.json, acre_fine.yaml, acre_coarse.yaml -> {out_root}")
    return report


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
