"""ACRE XML (instance-segmentation polygons) -> YOLO detection labels.

Produces, under ``paths.out_root``:
  images/<stem>.jpg            symlink to the raw image
  labels_fine/<stem>.txt       YOLO labels, 6 species classes  (PRIMARY)
  labels_coarse/<stem>.txt     YOLO labels, 2 classes crop/weed (SECONDARY)
  polygons/<stem>.json         original polygons + per-instance metadata
  annotations_master.json      flat per-box geometry+labels (no polygons; for analysis)
  convert_summary.json         counts, dropped boxes, OOV breakdown, sanity report

Design notes
------------
* Boxes are the tight min/max extent of the polygon (matching the official
  acre_xml_to_coco_obj_det.py), then clamped to the image and sanity-filtered.
* The FINE label uses the species in the XML <name> tag. The COARSE label uses
  the <class> tag (crop/weed/unknow->weed), but for the handful of *named* OOV
  species we trust the known biological group over a possibly noisy <class>.
* Out-of-vocabulary instances are dropped from the FINE labels (config policy)
  and counted; they are still represented in the COARSE labels.
* Everything is asserted / logged. Nothing fails silently.
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# allow `python src/data/convert.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import load_config  # noqa: E402


def parse_xml(path: Path) -> ET.ElementTree:
    """Parse with the same iso-8859-5 fallback the official script uses."""
    try:
        return ET.parse(path)
    except ET.ParseError:
        return ET.parse(path, parser=ET.XMLParser(encoding="iso-8859-5"))


def polygon_bbox(points):
    """Tight [x_min, y_min, x_max, y_max] around polygon vertices."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def main(cfg_path="configs/data.yaml"):
    cfg = load_config(cfg_path)
    raw_root = Path(cfg["paths"]["raw_root"])
    out_root = Path(cfg["paths"]["out_root"])

    fine_classes = cfg["fine_classes"]            # french -> {id, en, latin, group}
    coarse_map = cfg["coarse_map"]                # crop/weed -> 0/1
    tag_to_coarse = cfg["class_tag_to_coarse"]    # crop/weed/unknow -> crop/weed
    fine_group = {fr: d["group"] for fr, d in fine_classes.items()}
    oov_policy = cfg["oov_policy"]
    oov_known = cfg["oov_known_group"]
    boxcfg = cfg["box"]

    for sub in ["images", "labels_fine", "labels_coarse", "polygons"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    xml_files = sorted(raw_root.rglob("*.xml"))
    assert xml_files, f"no XML files under {raw_root}"

    # ---- sanity / stats accumulators -------------------------------------
    stem_seen = {}
    n_images = 0
    n_boxes_raw = 0                       # every clipping encountered
    n_boxes_fine = 0                      # written to fine labels
    n_boxes_coarse = 0                    # written to coarse labels
    fine_count = Counter()               # per fine class id
    coarse_count = Counter()             # per coarse class id
    oov_count = Counter()                # raw name -> count of OOV instances
    dropped = Counter()                  # reason -> count
    empty_images = []                    # stems with no boxes at all
    conflict_examples = []               # (stem, name, class_tag) species/group mismatch
    per_image_box_counts = []
    master = []                          # flat per-box records (no polygons)

    for xf in xml_files:
        tree = parse_xml(xf)
        root = tree.getroot()

        W = int(root.find("metadata/size/width").text)
        H = int(root.find("metadata/size/height").text)
        assert W > 0 and H > 0, f"bad size in {xf}"

        rel = xf.relative_to(raw_root)
        stem = rel.stem
        # filenames must be globally unique so a flat stem keying is safe
        if stem in stem_seen:
            raise RuntimeError(f"duplicate image stem {stem!r}: {xf} vs {stem_seen[stem]}")
        stem_seen[stem] = xf

        jpg = xf.with_suffix(".jpg")
        assert jpg.exists(), f"missing image for {xf}"

        n_images += 1
        fine_lines, coarse_lines, polys = [], [], []
        n_img_boxes = 0

        clippings = root.find("data/clippings")
        clips = list(clippings.iter("clipping")) if clippings is not None else []
        for clip in clips:
            n_boxes_raw += 1
            pts_node = clip.find("points")
            points = [(int(p.get("x")), int(p.get("y")))
                      for p in pts_node.iter("point")] if pts_node is not None else []
            if len(points) < 3:
                dropped["polygon_lt_3_vertices"] += 1
                continue

            x0, y0, x1, y1 = polygon_bbox(points)
            if boxcfg["clip_to_image"]:
                x0 = max(0, min(x0, W - 1)); x1 = max(0, min(x1, W - 1))
                y0 = max(0, min(y0, H - 1)); y1 = max(0, min(y1, H - 1))
            bw, bh = x1 - x0, y1 - y0
            if bw < boxcfg["min_side_px"] or bh < boxcfg["min_side_px"]:
                dropped["degenerate_side"] += 1
                continue
            if bw * bh < boxcfg["min_area_px"]:
                dropped["degenerate_area"] += 1
                continue

            name_node = clip.find("name")
            class_node = clip.find("class")
            raw_name = name_node.text if name_node is not None else None
            raw_class = class_node.text if class_node is not None else None
            tr = clip.find("trusted/isTrusted")
            is_trusted = (tr is not None and tr.text == "true")
            c = clip.find("center")
            center = None
            if c is not None and c.find("x") is not None and c.find("y") is not None \
                    and c.find("x").text is not None and c.find("y").text is not None:
                center = [int(c.find("x").text), int(c.find("y").text)]

            # ---- coarse group resolution (covers ALL instances) ----------
            if raw_name in fine_group:
                group = fine_group[raw_name]                  # known species
                # flag species/<class>-tag disagreement (annotation noise)
                tag_group = tag_to_coarse.get(raw_class)
                if tag_group is not None and tag_group != group and len(conflict_examples) < 30:
                    conflict_examples.append((stem, raw_name, raw_class))
            elif raw_name in oov_known and oov_known[raw_name] is not None:
                group = oov_known[raw_name]                   # named OOV w/ known group
            else:
                group = tag_to_coarse.get(raw_class)          # fall back to <class> tag
            if group is None:
                dropped["no_coarse_group"] += 1
                continue

            # normalized YOLO box (cx, cy, w, h)
            cx = ((x0 + x1) / 2) / W
            cy = ((y0 + y1) / 2) / H
            nw = bw / W
            nh = bh / H

            # ---- coarse label (always written) ---------------------------
            coarse_id = coarse_map[group]
            coarse_lines.append(f"{coarse_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            coarse_count[coarse_id] += 1
            n_boxes_coarse += 1

            # ---- fine label (only the 6 official species) ----------------
            fine_id = None
            if raw_name in fine_classes:
                fine_id = fine_classes[raw_name]["id"]
                fine_lines.append(f"{fine_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                fine_count[fine_id] += 1
                n_boxes_fine += 1
            else:
                oov_count[raw_name] += 1
                if oov_policy != "drop":
                    raise NotImplementedError("only oov_policy=drop is implemented")

            n_img_boxes += 1
            polys.append({
                "polygon": points, "n_vertices": len(points),
                "bbox_xyxy": [x0, y0, x1, y1],
                "raw_name": raw_name, "raw_class": raw_class,
                "fine_id": fine_id, "coarse_id": coarse_id,
                "center": center, "is_trusted": is_trusted,
            })
            master.append({
                "stem": stem, "img_w": W, "img_h": H,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "bw": bw, "bh": bh, "area_px": bw * bh,
                "fine_id": fine_id, "coarse_id": coarse_id,
                "raw_name": raw_name, "raw_class": raw_class,
            })

        # ---- write per-image artifacts (empty file is valid = negative) --
        (out_root / "labels_fine" / f"{stem}.txt").write_text("\n".join(fine_lines))
        (out_root / "labels_coarse" / f"{stem}.txt").write_text("\n".join(coarse_lines))
        (out_root / "polygons" / f"{stem}.json").write_text(json.dumps(polys))
        link = out_root / "images" / f"{stem}.jpg"
        if not link.exists():
            os.symlink(jpg.resolve(), link)

        per_image_box_counts.append(n_img_boxes)
        if n_img_boxes == 0:
            empty_images.append(stem)

    # ---- master annotations (flat, no polygons) --------------------------
    (out_root / "annotations_master.json").write_text(json.dumps(master))

    summary = {
        "n_images": n_images,
        "n_boxes_raw": n_boxes_raw,
        "n_boxes_fine": n_boxes_fine,
        "n_boxes_coarse": n_boxes_coarse,
        "fine_count_by_id": {cfg["fine_names"][i]: fine_count[i] for i in range(len(cfg["fine_names"]))},
        "coarse_count_by_id": {cfg["coarse_names"][i]: coarse_count[i] for i in range(len(cfg["coarse_names"]))},
        "oov_dropped_from_fine": dict(oov_count.most_common()),
        "oov_total": sum(oov_count.values()),
        "boxes_dropped": dict(dropped),
        "empty_images": empty_images,
        "n_empty_images": len(empty_images),
        "species_class_tag_conflicts_sample": conflict_examples,
        "box_count_per_image": {
            "min": min(per_image_box_counts), "max": max(per_image_box_counts),
            "mean": round(sum(per_image_box_counts) / len(per_image_box_counts), 2),
        },
    }
    (out_root / "convert_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- cross-validate against the official COCO conversion --------------
    coco_path = raw_root / "ACRE_COCO_annotations.json"
    if coco_path.exists():
        coco = json.load(open(coco_path))
        n_coco = len(coco["annotations"])
        summary["coco_crosscheck"] = {
            "coco_annotations": n_coco,
            "our_coarse_boxes": n_boxes_coarse,
            "delta": n_boxes_coarse - n_coco,
            "note": "small delta expected (we drop degenerate boxes; COCO keeps them)",
        }

    # ---- console report --------------------------------------------------
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"images: {n_images} | raw clippings: {n_boxes_raw} | "
          f"fine boxes: {n_boxes_fine} | coarse boxes: {n_boxes_coarse}")
    print(f"empty (background-only) images: {len(empty_images)} -> {empty_images}")
    print("\nFINE (6-class) instances:")
    for i, nm in enumerate(cfg["fine_names"]):
        print(f"  {i} {nm:14s} {fine_count[i]:6d}")
    print("\nCOARSE (2-class) instances:")
    for i, nm in enumerate(cfg["coarse_names"]):
        print(f"  {i} {nm:6s} {coarse_count[i]:6d}")
    print(f"\nOOV dropped from FINE ({sum(oov_count.values())} total, kept in COARSE):")
    for k, v in oov_count.most_common():
        print(f"  {str(k):14s} {v}")
    print(f"\nboxes dropped (sanity filters): {dict(dropped) or 'none'}")
    print(f"species/<class>-tag conflicts (sample): {conflict_examples[:5]}")
    if "coco_crosscheck" in summary:
        cc = summary["coco_crosscheck"]
        print(f"\nCOCO crosscheck: official={cc['coco_annotations']} "
              f"ours(coarse)={cc['our_coarse_boxes']} delta={cc['delta']}")
    print(f"\nwrote -> {out_root}")
    return summary


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
