"""Reversible, hash-checked migration of this workspace's legacy data layout.

Run plan, apply, verify, or rollback. Only paths recorded in the plan are moved;
existing destination files are never overwritten. Catalog can be refreshed later.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
PLAN = "data/manifests/layout_migration_20261127.json"
OLD_ROOTS = ("bike_dataset", "dataset_v2", "dataset_school", "incoming",
             "relabel/auto", "relabel/report", "scene_warp", "snap_out",
             "report_homography", "report_warp_snap", "violation_mask_out",
             "violation_mask_out2", "review.html")
NET = "data/network/used"
CAMPUS = "data/campus/used"
DATASETS = {"network_bicycle_v2": f"{NET}/annotated/bicycle_v2",
            "campus_school_v1": f"{CAMPUS}/annotated/school_v1"}


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path leaves workspace or names its root: {relative}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def destination(relative):
    parts = Path(relative).parts
    name = parts[0]
    tail = Path(*parts[1:]).as_posix()
    if name == "bike_dataset":
        if parts[1] == "labels":
            return f"{NET}/original_annotations/legacy_download/{tail}"
        if tail == "data.yaml":
            tail = "original_data.yaml"
        return f"{NET}/raw/legacy_download/{tail}"
    if name in ("dataset_v2", "dataset_school"):
        key = "network_bicycle_v2" if name == "dataset_v2" else "campus_school_v1"
        if name == "dataset_v2" and parts[1] == "preview":
            return "runs/previews/network_bicycle_v2/" + Path(*parts[2:]).as_posix()
        if name == "dataset_school" and parts[1] == "images" and "labels" in parts[2:-1]:
            return f"{CAMPUS}/annotation_work/legacy_accidental/{tail}"
        if Path(relative).suffix == ".npy" or Path(relative).suffix == ".cache":
            return f"runs/cache/datasets/{key}/{tail}"
        if relative.endswith(".bak"):
            return f"data/manifests/label_backups/{relative}"
        return f"{DATASETS[key]}/{tail}"
    if relative.startswith("incoming/单车违停检测/原始图片/"):
        return relative.replace("incoming/单车违停检测/原始图片/",
                                f"{CAMPUS}/raw/batch_20261125/images/", 1)
    if relative.startswith("incoming/单车违停检测/labels/原始图片/"):
        return relative.replace("incoming/单车违停检测/labels/原始图片/",
                                f"{CAMPUS}/annotation_work/batch_20261125/labels/", 1)
    if relative.startswith("incoming/alipan_2026-11-25/"):
        if parts[-1] == "manifest.json":
            return f"{CAMPUS}/raw/batch_20261125/download_manifest.json"
        if "labels" in parts:
            return f"{CAMPUS}/annotation_work/preview_labels/{parts[-1]}"
        return f"runs/previews/campus_batch_20261125/{parts[-1]}"
    if relative.startswith("incoming/"):
        # Unknown additions must not silently become 'unused' data.
        return "data/campus/usage_unknown/legacy_incoming/" + tail
    if relative.startswith("relabel/auto/"):
        tail = relative[len("relabel/auto/"):]
        if Path(relative).suffix.lower() in (".jpg", ".png", ".jpeg"):
            return "runs/previews/network_auto/" + tail
        return f"{NET}/pseudo_labels/yolo11x_v1/{tail}"
    if relative.startswith("relabel/report/"):
        return "runs/data_audit/network_legacy/" + relative[len("relabel/report/"):]
    if name == "review.html":
        return "runs/previews/network_bicycle_v2/review_legacy.html"
    return "runs/archive/" + relative


def build_plan(root):
    entries = []
    for old in OLD_ROOTS:
        path = safe_path(root, old)
        files = [path] if path.is_file() else sorted(path.rglob("*")) if path.exists() else []
        for file in files:
            if file.is_symlink():
                raise ValueError(f"Refusing linked source: {file}")
            if not file.is_file():
                continue
            source = file.relative_to(root).as_posix()
            target = destination(source)
            if safe_path(root, target).exists():
                raise FileExistsError(target)
            entries.append(dict(source=source, destination=target,
                                bytes=file.stat().st_size, sha256=sha256(file)))
    targets = [e["destination"].casefold() for e in entries]
    if len(targets) != len(set(targets)):
        raise ValueError("Migration has colliding destinations")
    return dict(version=1, workspace=str(root.resolve()), state="planned", files=entries)


def check_entry(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"Missing or changed file: {path}")


def transfer(root, plan, reverse=False):
    source_key, target_key = (("destination", "source") if reverse else ("source", "destination"))
    entries = list(reversed(plan["files"])) if reverse else plan["files"]
    # Validate every path and every file before the first mutation. A partial run
    # can be resumed: exactly one of source/destination must exist.
    for entry in entries:
        src, dst = safe_path(root, entry[source_key]), safe_path(root, entry[target_key])
        if src.exists() == dst.exists():
            raise ValueError(f"Expected one existing path: {src} / {dst}")
        check_entry(src if src.exists() else dst, entry)
    for entry in entries:
        src, dst = safe_path(root, entry[source_key]), safe_path(root, entry[target_key])
        if not src.exists():
            continue
        check_entry(src, entry)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        src.rename(dst)
    # Remove only empty containers, never recursively delete data.
    for old in OLD_ROOTS:
        path = safe_path(root, old)
        if path.is_dir():
            for folder in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if folder.is_dir():
                    try:
                        safe_path(root, folder.relative_to(root)).rmdir()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass


def verify(root, plan, reverse=False):
    key = "source" if reverse else "destination"
    for entry in plan["files"]:
        check_entry(safe_path(root, entry[key]), entry)
    return dict(files=len(plan["files"]), bytes=sum(e["bytes"] for e in plan["files"]),
                sha256_verified=True, location=key)


def scaffold(root):
    for source in ("network", "campus"):
        for suffix in ("unused/raw", "unused/annotated", "usage_unknown"):
            safe_path(root, f"data/{source}/{suffix}").mkdir(parents=True, exist_ok=True)
    for relative in DATASETS.values():
        for kind in ("images", "labels"):
            safe_path(root, relative + f"/{kind}/test").mkdir(parents=True, exist_ok=True)


def catalog(root):
    """Index actual images, not caches/previews; usage comes from experiment evidence."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows, datasets, role_by_hash = [], {}, defaultdict(set)
    for key, relative in DATASETS.items():
        base = safe_path(root, relative)
        splits, hashes = {}, defaultdict(list)
        for split in ("train", "val", "test"):
            images = sorted(p for p in (base / "images" / split).glob("*")
                            if p.is_file() and p.suffix.lower() in image_exts)
            splits[split] = len(images)
            for image in images:
                digest = sha256(image)
                label = base / "labels" / split / (image.stem + ".txt")
                lines = label.read_text(encoding="utf-8-sig").splitlines() if label.exists() else []
                row = dict(path=image.relative_to(root).as_posix(), sha256=digest,
                           source=key.split("_")[0], stage="annotated", usage="used",
                           split=split, label_path=label.relative_to(root).as_posix(),
                           label_state="boxes_present" if any(l.strip() for l in lines) else
                                       "empty_label_requires_provenance" if label.exists() else "missing",
                           parking_truth="not_verified", dataset=key)
                rows.append(row)
                hashes[digest].append(row["path"])
                role_by_hash[digest].add(split)
        conflicts = [paths for paths in hashes.values()
                     if len({Path(p).parent.name for p in paths}) > 1]
        datasets[key] = dict(path=relative, splits=splits, unique_images=len(hashes),
                             cross_split_duplicates=conflicts,
                             evidence="runs/bike11x_v1/args.yaml" if key.startswith("network")
                                      else "runs/school_v3/args.yaml",
                             independent_test_ready=False)
    for source, batch in (("network", "legacy_download"), ("campus", "batch_20261125")):
        base = safe_path(root, f"data/{source}/used/raw/{batch}/images")
        for image in sorted(base.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in image_exts:
                continue
            digest = sha256(image)
            roles = sorted(role_by_hash[digest])
            rows.append(dict(path=image.relative_to(root).as_posix(), sha256=digest,
                             source=source, stage="raw", usage="used", split="unassigned",
                             label_path="", label_state="not_a_training_dataset",
                             parking_truth="not_verified", dataset=batch,
                             known_training_roles=roles,
                             development_inference_used=source == "campus",
                             evidence="docs/2026-11-25_evidence.json" if source == "campus"
                                      else "relabel/report/dedupe.json (historical; now runs/data_audit/network_legacy)"))
    summary = dict(datasets=datasets, image_records=len(rows),
                   unique_images=len({r["sha256"] for r in rows}),
                   confirmed_unused_images=0, independent_test_images=0,
                   scope="Primary checkout only; nested repositories and worktrees are separate snapshots.",
                   counts=dict(Counter(f'{r["source"]}/{r["stage"]}' for r in rows)),
                   notes=["Used includes training, validation, inference and development inspection.",
                          "No old samples were randomly reassigned to an independent test set.",
                          "Empty incoming annotation files are unfinished work, not verified negatives.",
                          "Raw download's historical train/val folders are provenance, not current splits."])
    out = safe_path(root, "data/manifests")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "images.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "images.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["source", "stage", "usage", "split", "path", "sha256", "label_path",
                  "label_state", "parking_truth", "dataset"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply", "verify", "rollback", "catalog"))
    args = parser.parse_args()
    path = safe_path(ROOT, PLAN)
    if args.action == "plan":
        if path.exists():
            raise FileExistsError(f"Plan already exists; inspect {path}")
        plan = build_plan(ROOT)
        write_json(path, plan)
        print(json.dumps(dict(files=len(plan["files"]), plan=str(path)), ensure_ascii=False))
        return
    if args.action == "catalog":
        print(json.dumps(catalog(ROOT), ensure_ascii=False, indent=2))
        return
    plan = json.loads(path.read_text(encoding="utf-8"))
    if Path(plan["workspace"]).resolve() != ROOT:
        raise ValueError("Plan belongs to a different workspace")
    reverse = args.action == "rollback" or (args.action == "verify" and plan["state"] == "rolled_back")
    if args.action in ("apply", "rollback"):
        transfer(ROOT, plan, reverse=reverse)
        plan["state"] = "rolled_back" if reverse else "applied"
        write_json(path, plan)
        if not reverse:
            scaffold(ROOT)
            catalog(ROOT)
    print(json.dumps(verify(ROOT, plan, reverse=reverse), ensure_ascii=False))


if __name__ == "__main__":
    main()
