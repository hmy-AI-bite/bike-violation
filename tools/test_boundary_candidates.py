"""Geometry counterexamples and denominator checks for the frozen-mask study."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/20261127"))

import numpy as np
from boundary_candidates import (candidate_predictions, lower_envelope,
                                 ownership_diagnostics)
from evaluate_boundary_candidates import acceptance, metrics
from sam_prompts import center_point_prompt


class EvidenceTests(unittest.TestCase):
    def test_different_height_wheels_both_survive(self):
        mask=np.zeros((100,60),bool)
        mask[60:80,10:20]=True
        mask[20:40,40:50]=True
        lower=lower_envelope(mask)
        self.assertTrue(lower[79,15])
        self.assertTrue(lower[39,45])
        self.assertFalse(lower[:,25].any())
        self.assertFalse((lower & ~mask).any())

    def test_padding_does_not_change_evidence_thickness(self):
        mask=np.zeros((80,50),bool);mask[20:65,10:40]=True
        np.testing.assert_array_equal(lower_envelope(np.pad(mask,((17,40),(4,8))))[17:97,4:54],lower_envelope(mask))
        self.assertFalse(lower_envelope(np.zeros((10,10),bool)).any())
        with self.assertRaises(ValueError):
            lower_envelope(mask,0)

    def test_duplicate_masks_flagged_but_disjoint_masks_not(self):
        a=np.zeros((10,10),bool);a[2:5,2:5]=True
        b=np.zeros_like(a);b[7:9,7:9]=True
        boxes=[[0,.5,.5,1,1]]*3
        values=ownership_diagnostics(np.stack([a,a,b]),boxes)
        self.assertEqual(values[0]["max_mask_iou"],1)
        self.assertEqual(values[0]["neighbor_index"],1)
        self.assertEqual(values[2]["max_mask_iou"],0)
        self.assertIsNone(values[2]["neighbor_index"])
        self.assertTrue(all(v["box_leak"]==0 for v in values))

    def test_review_does_not_turn_ambiguity_into_compliance(self):
        mask=np.ones((10,10),bool)
        baseline={"verdict":"出界"}
        result=candidate_predictions(mask,~mask,baseline,{"max_mask_iou":.9,"box_leak":0})
        self.assertEqual(result["band"],"出界")
        self.assertEqual(result["band_with_review"],"无法判定")
        for verdict in ("无法判定","未标定"):
            result=candidate_predictions(mask,~mask,{"verdict":verdict},{"max_mask_iou":0,"box_leak":0})
            self.assertEqual(set(result.values()),{verdict})

    def test_ownership_diagnostics_do_not_mutate_masks(self):
        masks=np.ones((2,10,10),bool);saved=masks.copy()
        values=ownership_diagnostics(masks,[[0,.5,.5,.4,.4]]*2)
        np.testing.assert_array_equal(masks,saved)
        self.assertAlmostEqual(values[0]["box_leak"],.84)


class AccountingTests(unittest.TestCase):
    def row(self,reference,base,candidate):
        return dict(reference=reference,predictions={"band":base,"new":candidate})

    def test_abstention_and_unknown_reference_not_counted_as_correct(self):
        rows=[self.row("出界","出界","无法判定"),
              self.row("合规","出界","无法判定"),
              self.row("无法判定","出界","出界")]
        values=metrics(rows,"new")
        self.assertEqual(values["n"],2)
        self.assertEqual(values["resolved"],0)
        self.assertEqual(values["unresolved_positive"],1)
        self.assertEqual(values["unresolved_negative"],1)
        self.assertEqual(values["exact_agreement"],0)
        self.assertEqual(values["reference_unknown_outputs"],{"出界":1})
        self.assertFalse(acceptance(values,metrics(rows,"band"),{"image":{"new":values,"band":metrics(rows,"band")}},"new"))

    def test_tradeoff_between_images_fails_gate(self):
        image_a=[self.row("出界","合规","出界"),self.row("合规","出界","合规"),
                 self.row("合规","出界","合规")]
        image_b=[self.row("合规","合规","出界")]
        rows=image_a+image_b
        by_image={key:{name:metrics(group,name) for name in ("band","new")}
                  for key,group in (("a",image_a),("b",image_b))}
        self.assertFalse(acceptance(metrics(rows,"new"),metrics(rows,"band"),by_image,"new"))


class PromptTests(unittest.TestCase):
    def test_center_uses_processed_pixel_coordinates(self):
        box=np.array([10.,20.,70.,100.]);original=box.copy()
        prompt=center_point_prompt(box,1400,1867)
        np.testing.assert_array_equal(prompt["point_coords"],[[40,60]])
        np.testing.assert_array_equal(prompt["point_labels"],[1])
        np.testing.assert_array_equal(box,original)
        self.assertEqual(prompt["point_coords"].dtype,np.float32)

    def test_clipping_and_invalid_geometry(self):
        np.testing.assert_array_equal(center_point_prompt([-20,-10,10,6],20,20)["point_coords"],[[0,0]])
        for box in ([1,1,1,4],[0,0,float("nan"),1],[0,1,2]):
            with self.assertRaises(ValueError):
                center_point_prompt(box,20,20)


if __name__=="__main__":
    unittest.main()
