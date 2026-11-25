"""CPU 回归测试；不加载真实 YOLO/SAM 权重，不修改项目数据。"""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "yolo-lib"))
sys.path.insert(0, str(ROOT / "relabel"))

import numpy as np
from PIL import Image
import violation_mask
import violation
import train
from violation_rules import (bicycle_class_id, classify_mask, contact_band,
                             count_verdicts, resolve_zones)

POLY = [[0, 0], [1, 0], [1, 1], [0, 1]]


class RuleTests(unittest.TestCase):
    def test_coco_and_custom_classes(self):
        self.assertEqual(bicycle_class_id({0: "person", 1: "bicycle"}), 1)
        self.assertEqual(bicycle_class_id(["bicycle", "parking_area"]), 0)
        self.assertEqual(bicycle_class_id({"0": "bicycle"}), 0)
        for names in (["car"], ["bicycle", "bicycle"]):
            with self.assertRaises(ValueError):
                bicycle_class_id(names)

    def test_empty_override_and_missing_key(self):
        scene = dict(allow=POLY, deny=[POLY], images={
            "clear": dict(allow=[], deny=[]), "no_deny": dict(deny=[])})
        self.assertEqual(resolve_zones(scene, "clear"), ([], []))
        self.assertEqual(resolve_zones(scene, "no_deny"), (POLY, []))
        self.assertEqual(resolve_zones(scene, "missing"), (POLY, [POLY]))
        self.assertEqual(violation.load_scene({"s": scene}, "s", "clear"),
                         dict(allow=[], deny=[]))

    def test_invalid_polygon_rejected(self):
        for poly in ([[0, 0], [1, 1]], [[0, 0], [0.5, 0.5], [1, 1]],
                     [[0, 0], [1, 0], [float("nan"), 1]],
                     [[0, 0], [2, 0], [0, 1]]):
            with self.assertRaises(ValueError):
                resolve_zones({"allow": poly}, "x")

    def test_contact_band_uses_target_height(self):
        small = np.zeros((100, 30), bool)
        large = np.zeros((1000, 30), bool)
        small[20:40, 5:15] = True
        large[20:40, 5:15] = True
        self.assertEqual(int(contact_band(small).sum()), 20)
        self.assertEqual(int(contact_band(large).sum()), 20)
        self.assertFalse(contact_band(np.zeros((10, 10), bool)).any())

    def classify(self, cols, **kwargs):
        mask = np.ones((10, 10), bool)
        allow = np.zeros_like(mask)
        allow[:, :cols] = True
        return classify_mask(mask, allow, np.zeros_like(mask),
                             has_zone=True, score=0.9, min_area=1, **kwargs)

    def test_verdict_boundaries(self):
        for cols, verdict in ((10, "合规"), (9, "合规"), (5, "压线"),
                              (1, "出界"), (0, "出界")):
            with self.subTest(cols=cols):
                self.assertEqual(self.classify(cols)["verdict"], verdict)

    def test_deny_priority_and_tolerance(self):
        m = np.ones((10, 10), bool)
        deny = np.zeros_like(m)
        deny[0, :5] = True
        result = classify_mask(m, m, deny, has_zone=True, score=0.9, min_area=1)
        self.assertEqual(result["verdict"], "合规")
        deny[0, 5] = True
        result = classify_mask(m, m, deny, has_zone=True, score=0.9, min_area=1)
        self.assertEqual(result["verdict"], "占道")

    def test_failures_never_compliant(self):
        full = np.ones((10, 10), bool)
        for mask, score, threshold, reason in (
                (np.zeros_like(full), 0.9, 0, "small_mask"),
                (full, 0.9, 0, "small_mask"),
                (full, 0.3, 0.5, "low_score"),
                (full, float("nan"), 0, "invalid_score")):
            result = classify_mask(mask, full, ~full, has_zone=True,
                                   score=score, min_score=threshold)
            self.assertEqual(result["verdict"], "无法判定")
            self.assertEqual(result["reason"], reason)
            self.assertEqual(result["contact_in"], -1)
        result = classify_mask(full, full, ~full, has_zone=False, score=0)
        self.assertEqual(result["verdict"], "未标定")

    def test_summary_accounts_for_unknown(self):
        counts = count_verdicts([{"verdict": v} for v in
                                ("合规", "出界", "无法判定", "未标定")])
        self.assertEqual(sum(counts.values()), 4)
        self.assertEqual(counts["合规"], 1)


class PipelineTests(unittest.TestCase):
    def exercise(self, *, allow=POLY, detections=2):
        class Tensor:
            def __init__(self, array):
                self.array = array
            def cpu(self):
                return self
            def numpy(self):
                return self.array
        calls = dict(predict=0, encode=0, masks=0)
        class Detector:
            names = {0: "bicycle", 1: "parking_area"}
            def __init__(self, weights):
                pass
            def predict(self, image, **kwargs):
                self_test.assertEqual(kwargs["classes"], [0])
                np.testing.assert_array_equal(image[0, 0], [0, 0, 255])
                calls["predict"] += 1
                boxes = np.array([[2, 2, 12, 12], [15, 15, 16, 16]], float)[:detections]
                return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=Tensor(boxes)))]
        class Predictor:
            def __init__(self, model):
                pass
            def set_image(self, image):
                np.testing.assert_array_equal(image[0, 0], [255, 0, 0])
                calls["encode"] += 1
            def predict(self, box, **kwargs):
                calls["masks"] += 1
                mask = np.zeros((20, 20), bool)
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = True
                return mask[None], np.array([0.9]), None
        self_test = self
        modules = {"ultralytics": types.SimpleNamespace(YOLO=Detector),
                   "sam2": types.ModuleType("sam2"),
                   "sam2.build_sam": types.SimpleNamespace(build_sam2=lambda *a, **k: object()),
                   "sam2.sam2_image_predictor": types.SimpleNamespace(SAM2ImagePredictor=Predictor)}
        with tempfile.TemporaryDirectory(prefix="_test_violation_", dir=ROOT / "tools") as temp:
            base = Path(temp)
            Image.new("RGB", (20, 20), (255, 0, 0)).save(base / "input.png")
            (base / "scenes.json").write_text(json.dumps({"s": {"allow": allow}}), encoding="utf-8")
            out = base / "out"
            out.mkdir()
            (out / "per_bike.csv").write_text("stale data", encoding="utf-8")
            argv = ["violation_mask.py", "--source", str(base / "input.png"),
                    "--scenes", str(base / "scenes.json"), "--scene", "s",
                    "--out", str(out), "--min-mask-area", "5"]
            output = io.StringIO()
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                violation_mask.main()
            with (out / "per_bike.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue((out / "input.png").exists())
            return rows, calls, output.getvalue()

    def test_failed_mask_kept_in_csv_and_summary(self):
        rows, calls, output = self.exercise()
        self.assertEqual([r["verdict"] for r in rows], ["合规", "无法判定"])
        self.assertEqual(rows[1]["reason"], "small_mask")
        self.assertIn("合规 1 / 违停 0 / 无法判定 1 / 检测 2", output)
        self.assertNotIn("框的误报", output)
        self.assertEqual(calls["masks"], 2)

    def test_no_zone_does_not_run_sam(self):
        rows, calls, _ = self.exercise(allow=[])
        self.assertEqual([r["verdict"] for r in rows], ["未标定", "未标定"])
        self.assertEqual(calls["encode"], 0)
        self.assertEqual(calls["masks"], 0)

    def test_zero_detections_clears_stale_csv(self):
        rows, calls, _ = self.exercise(detections=0)
        self.assertEqual(rows, [])
        self.assertEqual(calls["encode"], 0)


class TrainingTests(unittest.TestCase):
    def test_nonfinite_losses_and_metrics(self):
        for metrics, losses in (({"val/box_loss": float("nan")}, {}),
                                ({}, {"box_loss": float("inf")})):
            trainer = types.SimpleNamespace(epoch=36, metrics=metrics, tloss=losses,
                                            label_loss_items=lambda x, prefix: x)
            with self.assertRaisesRegex(RuntimeError, "37"):
                train.check_finite_metrics(trainer)
        train.check_finite_metrics(types.SimpleNamespace(epoch=0, metrics={"mAP": 0}, tloss=None))

    def test_default_and_explicit_amp(self):
        for options, expected in (([], True), (["--amp"], True), (["--no-amp"], False)):
            captured = {}
            class Model:
                def __init__(self, weights):
                    pass
                def add_callback(self, event, callback):
                    captured[event] = callback
                def train(self, **kwargs):
                    captured.update(kwargs)
            with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=Model)}), patch.object(sys, "argv", ["train.py"] + options):
                train.main()
            self.assertEqual(captured["amp"], expected)
            self.assertEqual(captured["lr0"], 0.002)
            self.assertEqual(captured["workers"], 0)
            self.assertIn("on_fit_epoch_end", captured)
            self.assertEqual(Path(captured["data"]), ROOT / "dataset_v2/data.yaml")


if __name__ == "__main__":
    unittest.main()
