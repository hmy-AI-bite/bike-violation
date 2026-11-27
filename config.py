"""Shared project paths and launchers. Environment variables override .env.

Relative configured paths are resolved against this file, never the current cwd.
This module uses only the standard library until load_font() is called.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def load_env(path):
    """Read literal KEY=VALUE settings; do not execute or expand shell syntax."""
    path = Path(path)
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.strip().isidentifier():
            raise ValueError(f"Invalid setting at {path.name}:{number}; expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


env_file = Path(os.environ.get("BICYCLE_ENV_FILE", ROOT / ".env")).expanduser()
load_env(env_file if env_file.is_absolute() else ROOT / env_file)


def env_path(name, default):
    value = Path(os.environ.get(name) or default).expanduser()
    return value if value.is_absolute() else ROOT / value


DATA_ROOT = env_path("BICYCLE_DATA_ROOT", ROOT / "data")
OUTPUT_ROOT = env_path("BICYCLE_OUTPUT_ROOT", ROOT)
WEIGHTS_DIR = env_path("BICYCLE_WEIGHTS_DIR", ROOT / "relabel/weights")
YOLO_LIB = env_path("BICYCLE_YOLO_LIB", ROOT / "yolo-lib")
SAM2_REPO = env_path("BICYCLE_SAM2_REPO", ROOT / "vendor/sam2")
SAM2_CHECKPOINT = env_path("BICYCLE_SAM2_CHECKPOINT", WEIGHTS_DIR / "sam2.1_hiera_tiny.pt")
SAM2_CONFIG = os.environ.get("BICYCLE_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
DATA_ALIASES = {
    "bike_dataset": "network/used/raw/legacy_download",
    "dataset_v2": "network/used/annotated/bicycle_v2",
    "dataset_school": "campus/used/annotated/school_v1",
    "school_source": "campus/used/raw/legacy_school_source",
    "incoming/单车违停检测/原始图片": "campus/used/raw/batch_20261125/images",
    "incoming/单车违停检测/labels/原始图片": "campus/used/annotation_work/batch_20261125/labels",
}


def _mapped(relative, aliases):
    normalized = str(relative).replace("\\", "/").rstrip("/")
    for old, new in sorted(aliases.items(), key=lambda item: -len(item[0])):
        if normalized == old or normalized.startswith(old + "/"):
            return new + normalized[len(old):]
    return normalized


def data_path(relative=""):
    mapped = DATA_ROOT / _mapped(relative, DATA_ALIASES)
    # During migration/rollback, read the original checkout if its mapped
    # destination has not arrived yet. Explicit external data roots stay literal.
    old = ROOT / relative
    if DATA_ROOT == ROOT / "data" and old.exists():
        available = mapped.exists()
        if available and mapped.is_dir():
            available = any(p.is_file() for p in mapped.rglob("*"))
        if not available:
            return str(old)
    return str(mapped)


def output_path(relative=""):
    normalized = str(relative).replace("\\", "/")
    if normalized == "relabel/auto" or normalized.startswith("relabel/auto/"):
        mapped = DATA_ROOT / ("network/used/pseudo_labels/yolo11x_v1" + normalized[len("relabel/auto"):])
    else:
        mapped = OUTPUT_ROOT / _mapped(normalized, {
            "relabel/report": "runs/data_audit/network_legacy",
            "review.html": "runs/previews/network_bicycle_v2/review.html",
            **{name: "runs/archive/" + name for name in
               ("scene_warp", "snap_out", "report_homography", "report_warp_snap",
                "violation_mask_out", "violation_mask_out2")},
        })
    old = OUTPUT_ROOT / relative
    if not mapped.exists() and old.exists():
        return str(old)
    return str(mapped)


NETWORK_DATASET = Path(data_path("dataset_v2"))
CAMPUS_DATASET = Path(data_path("dataset_school"))
CAMPUS_RAW_IMAGES = Path(data_path("incoming/单车违停检测/原始图片"))
CAMPUS_ANNOTATIONS = Path(data_path("incoming/单车违停检测/labels/原始图片"))
SCHOOL_SOURCE = env_path("BICYCLE_SCHOOL_SOURCE", data_path("school_source"))
SCHOOL_IMAGES = env_path("BICYCLE_SCHOOL_IMAGES", CAMPUS_DATASET / "images/train")
SCENES = env_path("BICYCLE_SCENES", CAMPUS_DATASET / "scenes.json")
SCENE = os.environ.get("BICYCLE_SCENE", "school_row")
DETECT_WEIGHTS = env_path("BICYCLE_DETECT_WEIGHTS", OUTPUT_ROOT / "runs/school_v3/weights/best.pt")
PYTHON = os.environ.get("BICYCLE_PYTHON") or sys.executable
GUI_PYTHON = os.environ.get("BICYCLE_GUI_PYTHON") or PYTHON


def resolve_project_path(value):
    """Resolve old paths in saved experiment metadata without rewriting evidence."""
    path = Path(value)
    try:
        relative = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    except ValueError:
        return str(path)
    if _mapped(relative, DATA_ALIASES) != relative:
        return data_path(relative)
    return output_path(relative)


def annotation_paths(folder):
    """Keep the imported campus images separate from unfinished annotations."""
    root = Path(folder).resolve()
    try:
        parts = root.relative_to(DATA_ROOT.resolve()).parts
    except ValueError:
        parts = ()
    if len(parts) >= 4 and parts[0] in ("campus", "network") and parts[2] == "raw":
        images = root / "images" if (root / "images").is_dir() else root
        tail = parts[5:] if len(parts) >= 5 and parts[4] == "images" else ()
        labels = DATA_ROOT / parts[0] / parts[1] / "annotation_work" / parts[3] / "labels"
        return images, labels.joinpath(*tail)
    if (root / "images").is_dir():
        return root / "images", root / "labels"
    if root.name.lower() == "images":
        return root, root.parent / "labels"
    return root, root / "labels"


def weight_path(filename):
    return str(WEIGHTS_DIR / filename)


def school_image(filename):
    """Locate historical reference images across the configured school splits."""
    candidates = [SCHOOL_SOURCE / "images" / filename, SCHOOL_IMAGES / filename,
                  CAMPUS_DATASET / "images/val" / filename]
    return str(next((p for p in candidates if p.is_file()), candidates[0]))


def activate():
    """Make direct script execution work with installed or local dependencies."""
    for path in (ROOT / "tools", ROOT / "relabel", YOLO_LIB, SAM2_REPO):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("YOLO_CONFIG_DIR", output_path("relabel/cfg"))
    os.environ.setdefault("MPLCONFIGDIR", output_path("relabel/cfg/mpl"))


def load_font(size):
    from PIL import ImageFont
    configured = os.environ.get("BICYCLE_FONT_PATH")
    candidates = [str(env_path("BICYCLE_FONT_PATH", configured))] if configured else []
    if os.environ.get("WINDIR"):
        candidates.append(str(Path(os.environ["WINDIR"]) / "Fonts/msyh.ttc"))
    candidates.append("DejaVuSans.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def launch(command, extra):
    commands = {
        "labeler": ("tools/labeler.py", [data_path("dataset_v2")]),
        "zones": ("tools/zone_tool.py", [str(SCHOOL_IMAGES), str(SCENES), SCENE]),
        "detect": ("relabel/violation_mask.py", ["--source", str(SCHOOL_IMAGES),
            "--scenes", str(SCENES), "--scene", SCENE, "--weights", str(DETECT_WEIGHTS),
            "--out", output_path("runs/parking_detection")]),
        "train-demo": ("relabel/train.py", ["--data", data_path("dataset_school/data.yaml"),
            "--weights", weight_path("yolo11s.pt"), "--imgsz", "1280", "--epochs", "20",
            "--batch", "8", "--lr0", "0.002", "--workers", "0", "--amp", "--name", "demo_visible"]),
    }
    script, defaults = commands[command]
    activate()
    # Positional GUI arguments replace defaults; argparse options can override defaults.
    args = extra if extra and command in ("labeler", "zones") else defaults + extra
    interpreter = GUI_PYTHON if command in ("labeler", "zones") else PYTHON
    result = subprocess.run([interpreter, str(ROOT / script), *args], cwd=ROOT)
    if result.returncode == 0 and command == "detect" and os.name == "nt":
        os.startfile(output_path("runs/parking_detection"))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("show", help="Show resolved paths")
    run = sub.add_parser("run", help="Run a configured launcher")
    run.add_argument("command", choices=["labeler", "zones", "detect", "train-demo"])
    run.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "run":
        return launch(args.command, args.extra)
    print(json.dumps({key: str(value) for key, value in globals().items()
                     if key.isupper() and isinstance(value, (str, Path))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
