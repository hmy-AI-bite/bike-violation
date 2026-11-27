"""Back up all data/ and runs/ files to another drive and verify every byte.

Creates a new directory, never overwrites an existing backup. Requires only
Python's standard library; includes SHA-256 manifests and annotation counts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = "data/campus/used/annotation_work/ground_truth_v1/labels/image"


def digest(stream):
    value = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        value.update(chunk)
    return value.hexdigest()


def inventory():
    files = []
    for name in ("data", "runs"):
        folder = ROOT / name
        if not folder.is_dir():
            raise RuntimeError(f"Missing source: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink() or path.is_junction():
                raise RuntimeError(f"Linked path needs explicit handling: {path}")
            if path.is_file():
                stat = path.stat()
                files.append((path, stat.st_size, stat.st_mtime_ns))
    return files


def annotation_summary(documents):
    counts = Counter()
    images = {}
    for name, doc in documents.items():
        states = Counter(obj["parking_state"] for obj in doc["objects"])
        images[name] = {"vehicles": len(doc["objects"]), "states": dict(states)}
        counts.update(states)
    if len(images) != 5 or sum(counts.values()) != 157:
        raise RuntimeError(f"Expected 5 authoritative reviews / 157 vehicles: {images}")
    return {"review_files": len(images), "vehicles": sum(counts.values()),
            "states": dict(counts), "images": images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="New backup directory on a different drive")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.drive.lower() == ROOT.drive.lower():
        parser.error("Choose a different drive from the project")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tag_commit = subprocess.check_output(
        ["git", "rev-parse", args.tag + "^{commit}"], cwd=ROOT, text=True).strip()
    if commit != tag_commit:
        raise RuntimeError("Tag must point at the current baseline commit")
    files = inventory()
    reviews = {p.relative_to(ROOT).as_posix(): json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((ROOT / REVIEWS).glob("*.review.json"))}
    human = annotation_summary(reviews)
    target.mkdir(parents=True, exist_ok=False)
    archive = target / "data_and_runs.zip"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(ROOT), "git_commit": commit, "git_tag": args.tag,
                "annotations": human, "files": [], "totals": {}}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as output:
        for number, (path, size, modified) in enumerate(files, 1):
            relative = path.relative_to(ROOT).as_posix()
            sha = hashlib.sha256()
            with path.open("rb") as source, output.open(relative, "w", force_zip64=True) as dest:
                while chunk := source.read(1024 * 1024):
                    sha.update(chunk)
                    dest.write(chunk)
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (size, modified):
                raise RuntimeError(f"Source changed during backup: {relative}")
            manifest["files"].append({"path": relative, "bytes": size,
                                      "sha256": sha.hexdigest()})
            if number % 500 == 0:
                print(f"Archived {number}/{len(files)} files", flush=True)
        manifest["totals"] = {
            name: {"files": sum(e["path"].startswith(name + "/") for e in manifest["files"]),
                   "bytes": sum(e["bytes"] for e in manifest["files"]
                                if e["path"].startswith(name + "/"))}
            for name in ("data", "runs")}
        output.writestr("BACKUP_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Verifying every archived file and the five authoritative reviews", flush=True)
    with zipfile.ZipFile(archive) as saved:
        expected = {entry["path"] for entry in manifest["files"]} | {"BACKUP_MANIFEST.json"}
        if set(saved.namelist()) != expected or len(saved.namelist()) != len(expected):
            raise RuntimeError("Archive file list differs from source inventory")
        for entry in manifest["files"]:
            with saved.open(entry["path"]) as stream:
                if digest(stream) != entry["sha256"]:
                    raise RuntimeError(f"Archive SHA-256 mismatch: {entry['path']}")
            if saved.getinfo(entry["path"]).file_size != entry["bytes"]:
                raise RuntimeError(f"Archive size mismatch: {entry['path']}")
        restored = {name: json.loads(saved.read(name)) for name in reviews}
        if restored != reviews or annotation_summary(restored) != human:
            raise RuntimeError("Archived manual annotation contents differ")
    if inventory() != files:
        raise RuntimeError("Source file inventory changed during backup")
    # Re-read the source too, so unchanged file sizes cannot hide a mismatch.
    for entry in manifest["files"]:
        with (ROOT / entry["path"]).open("rb") as stream:
            if digest(stream) != entry["sha256"]:
                raise RuntimeError(f"Source changed: {entry['path']}")
    with archive.open("rb") as stream:
        archive_hash = digest(stream)
    report = {"status": "verified", "archive": str(archive),
              "archive_bytes": archive.stat().st_size, "archive_sha256": archive_hash,
              "git_commit": commit, "git_tag": args.tag,
              "files_verified": len(files), "totals": manifest["totals"],
              "annotations": human}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "SHA256SUMS.txt").write_text(f"{archive_hash}  {archive.name}\n", encoding="ascii")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
