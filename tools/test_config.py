"""Verify env precedence, relocation, and launch arguments without loading models."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config


class ConfigTests(unittest.TestCase):
    def test_relocated_repository_and_environment_precedence(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tools") as temp:
            copied = Path(temp) / "project with spaces"
            copied.mkdir()
            shutil.copyfile(ROOT / "config.py", copied / "config.py")
            (copied / ".env").write_text(
                'BICYCLE_DATA_ROOT="local data"\nBICYCLE_OUTPUT_ROOT=local outputs\n', encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if not k.startswith("BICYCLE_")}
            env["BICYCLE_OUTPUT_ROOT"] = "environment outputs"
            result = subprocess.run([sys.executable, str(copied / "config.py"), "show"],
                                    cwd=ROOT / "relabel", env=env, capture_output=True,
                                    text=True, encoding="utf-8", check=True)
            data = json.loads(result.stdout)
            self.assertEqual(Path(data["ROOT"]), copied)
            self.assertEqual(Path(data["DATA_ROOT"]), copied / "local data")
            self.assertEqual(Path(data["OUTPUT_ROOT"]), copied / "environment outputs")
            self.assertEqual(Path(data["WEIGHTS_DIR"]), copied / "relabel/weights")

    def test_custom_env_file_resolves_from_repository(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tools") as temp:
            copied = Path(temp)
            shutil.copyfile(ROOT / "config.py", copied / "config.py")
            (copied / "settings.env").write_text('BICYCLE_SCENE=custom_scene\n', encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if not k.startswith("BICYCLE_")}
            env["BICYCLE_ENV_FILE"] = "settings.env"
            result = subprocess.run([sys.executable, str(copied / "config.py"), "show"],
                                    cwd=ROOT / "relabel", env=env, capture_output=True,
                                    text=True, encoding="utf-8", check=True)
            self.assertEqual(json.loads(result.stdout)["SCENE"], "custom_scene")

    def test_env_values_are_literal_and_do_not_override_environment(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tools") as temp:
            file = Path(temp) / ".env"
            file.write_text('KEEP=from_file\nLITERAL="$(never_execute) %NAME%"\n', encoding="utf-8")
            with patch.dict(os.environ, {"KEEP": "existing"}, clear=True):
                config.load_env(file)
                self.assertEqual(os.environ["KEEP"], "existing")
                self.assertEqual(os.environ["LITERAL"], "$(never_execute) %NAME%")

    def test_malformed_config_reports_line(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tools") as temp:
            file = Path(temp) / "bad.env"
            file.write_text("# comment\nnot a setting\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.env:2"):
                config.load_env(file)

    def test_gui_launcher_uses_gui_interpreter_and_keeps_spaces(self):
        fake = types.SimpleNamespace(returncode=0)
        with patch.object(config.subprocess, "run", return_value=fake) as run:
            self.assertEqual(config.launch("zones", ["images with spaces", "zones.json", "scene"]), 0)
        args = run.call_args.args[0]
        self.assertEqual(args, [config.GUI_PYTHON, str(ROOT / "tools/zone_tool.py"),
                                "images with spaces", "zones.json", "scene"])
        self.assertEqual(run.call_args.kwargs["cwd"], ROOT)

    def test_training_launcher_propagates_failure_and_override(self):
        fake = types.SimpleNamespace(returncode=7)
        with patch.object(config.subprocess, "run", return_value=fake) as run:
            self.assertEqual(config.launch("train-demo", ["--epochs", "1"]), 7)
        args = run.call_args.args[0]
        self.assertEqual(args[0], config.PYTHON)
        self.assertEqual(args[-2:], ["--epochs", "1"])
        self.assertIn(config.data_path("dataset_school/data.yaml"), args)

    def test_organized_paths_and_historical_metadata_resolve(self):
        with patch.object(config, "DATA_ROOT", ROOT / "tools/nonexistent-data-root"):
            self.assertEqual(Path(config.data_path("dataset_v2/data.yaml")),
                             config.DATA_ROOT / "network/used/annotated/bicycle_v2/data.yaml")
            old = str(ROOT / "incoming/单车违停检测/原始图片/点位A/a.jpg")
            self.assertEqual(Path(config.resolve_project_path(old)), config.DATA_ROOT /
                             "campus/used/raw/batch_20261125/images/点位A/a.jpg")

    def test_raw_annotation_paths_never_write_back_into_raw_data(self):
        raw = config.DATA_ROOT / "campus/unused/raw/new_batch/images/angle1"
        images, labels = config.annotation_paths(raw)
        self.assertEqual(images, raw.resolve())
        self.assertEqual(labels, config.DATA_ROOT / "campus/unused/annotation_work/new_batch/labels/angle1")

    def test_rollback_ignores_empty_destination_containers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tools") as temp:
            root = Path(temp)
            old = root / "dataset_v2"
            old.mkdir()
            (old / "data.yaml").write_text("train: images/train\n")
            (root / "data/network/used/annotated/bicycle_v2/images/test").mkdir(parents=True)
            with patch.object(config, "ROOT", root), patch.object(config, "DATA_ROOT", root / "data"):
                self.assertEqual(Path(config.data_path("dataset_v2")), old)


if __name__ == "__main__":
    unittest.main()
