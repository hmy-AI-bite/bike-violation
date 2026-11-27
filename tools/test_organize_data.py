"""Exercise migration safety and recovery on temporary data only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import organize_data as migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="data_migration_test_", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def put(self, name, content=b"sample"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_roundtrip_preserves_original_images_labels_and_metadata(self):
        files = ["bike_dataset/images/train/a.jpg", "bike_dataset/labels/train/a.txt",
                 "dataset_v2/images/val/b.jpg", "dataset_v2/labels/val/b.review.json",
                 "dataset_school/images/train/a.npy",
                 "dataset_school/images/train/labels/a.txt",
                 "incoming/单车违停检测/labels/原始图片/点位A/a.txt"]
        for i, name in enumerate(files):
            self.put(name, str(i).encode())
        plan = migration.build_plan(self.root)
        migration.transfer(self.root, plan)
        self.assertEqual(migration.verify(self.root, plan)["files"], len(files))
        self.assertFalse((self.root / "dataset_v2").exists())
        self.assertTrue((self.root / "data/network/used/original_annotations/legacy_download/labels/train/a.txt").exists())
        migration.transfer(self.root, plan, reverse=True)
        migration.verify(self.root, plan, reverse=True)
        self.assertTrue(all((self.root / name).exists() for name in files))

    def test_changed_source_fails_before_any_move(self):
        self.put("dataset_v2/images/train/a.jpg")
        b = self.put("dataset_v2/images/train/b.jpg")
        plan = migration.build_plan(self.root)
        b.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            migration.transfer(self.root, plan)
        self.assertTrue((self.root / "dataset_v2/images/train/a.jpg").exists())

    def test_partial_migration_can_resume_and_apply_twice(self):
        self.put("dataset_v2/images/train/a.jpg")
        self.put("dataset_v2/images/train/b.jpg", b"b")
        plan = migration.build_plan(self.root)
        entry = plan["files"][0]
        destination = self.root / entry["destination"]
        destination.parent.mkdir(parents=True)
        (self.root / entry["source"]).rename(destination)
        migration.transfer(self.root, plan)
        migration.transfer(self.root, plan)
        migration.verify(self.root, plan)

    def test_collision_and_escape_are_rejected(self):
        self.put("dataset_v2/images/train/a.jpg")
        plan = migration.build_plan(self.root)
        self.put(plan["files"][0]["destination"], b"do not overwrite")
        with self.assertRaises(ValueError):
            migration.transfer(self.root, plan)
        for invalid in ("../outside", str(self.root), str(self.root.parent / "elsewhere")):
            with self.assertRaises(ValueError):
                migration.safe_path(self.root, invalid)

    def test_catalog_flags_leakage_without_reassigning_splits(self):
        dataset = migration.DATASETS["campus_school_v1"]
        for split in ("train", "val"):
            self.put(f"{dataset}/images/{split}/a.jpg", b"identical image")
            self.put(f"{dataset}/labels/{split}/a.txt", b"0 .5 .5 .2 .2\n")
        migration.scaffold(self.root)
        summary = migration.catalog(self.root)
        stats = summary["datasets"]["campus_school_v1"]
        self.assertEqual(stats["splits"], {"train": 1, "val": 1, "test": 0})
        self.assertEqual(len(stats["cross_split_duplicates"]), 1)
        self.assertEqual(summary["confirmed_unused_images"], 0)
        self.assertEqual(summary["independent_test_images"], 0)
        self.assertEqual(len((self.root / "data/manifests/images.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
