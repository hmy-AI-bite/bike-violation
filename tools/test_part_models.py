"""Counterexamples for part association and proof that lineage export is read-only."""
from pathlib import Path
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/20261127'))
import numpy as np
import torch
from compare_part_models import select_unique,mask_owner,box_iou,candidate_metrics
from run_part_inference import attach_parent_export


class AssociationTests(unittest.TestCase):
    def test_identical_overlapping_cars_are_not_arbitrarily_assigned(self):
        mask=np.ones((10,10),bool)
        self.assertIsNone(mask_owner(mask,[[0,0,10,10],[0,0,10,10]])[0])
        self.assertEqual(mask_owner(mask,[[0,0,10,10],[0,0,2,2]])[0],0)

    def test_out_of_frame_box_does_not_wrap_slices(self):
        mask=np.ones((10,10),bool)
        self.assertIsNone(mask_owner(mask,[[-20,-20,-1,-1]])[0])
        self.assertIsNone(mask_owner(mask,[[12,12,30,30]])[0])

    def test_iou_and_uncertain_overlap(self):
        self.assertEqual(box_iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(box_iou([0,0,1,1],[2,2,3,3]),0)
        self.assertIsNone(select_unique([.75,.7],.5,.1)[0])
        self.assertIsNone(select_unique([.2],.5,.1)[0])

    def test_uncovered_positive_stays_in_denominator(self):
        rows=[dict(reference='出界',flag=False,assigned_wheels=0),
              dict(reference='合规',flag=True,assigned_wheels=1),
              dict(reference='无法判定',flag=True,assigned_wheels=1)]
        values=candidate_metrics(rows,'flag')
        self.assertEqual(values['positive_denominator'],1)
        self.assertEqual(values['without_candidate'],1)
        self.assertEqual(values['false_candidate'],1)
        self.assertEqual(values['reference_unknown_candidates'],1)
        self.assertEqual(values['vehicles_with_assigned_wheel'],1)


class LineageTests(unittest.TestCase):
    def test_topk_export_preserves_prediction_and_parent_index(self):
        # Object queries 2 and 0 are parents; 2 part queries per parent.
        obj={'pred_logits':torch.tensor([[[3.],[1.],[5.]]]),
             'pred_boxes':torch.arange(12).reshape(1,3,4).float()}
        part={'pred_logits':torch.tensor([[[1.],[2.],[7.],[4.]]])}
        expected={'pred_logits':torch.tensor([[[5.,-100.],[-100.,7.],[-100.,4.]]])}
        class Core:
            topk_object_queries_num=2
            num_part_queries=2
            def hierarchical_topk_outputs(self,*args):
                return expected
        class Model:
            partglee=Core()
            test_topk_per_image=3
            def instance_inference(self,mask_cls,mask_pred,boxes,task):
                return types.SimpleNamespace(original=mask_cls.clone())
        model=Model();attach_parent_export(model)
        actual=model.partglee.hierarchical_topk_outputs('task','objects','parts',[],obj,part,3,1,2,None,None)
        self.assertIs(actual,expected)
        result=model.instance_inference(actual['pred_logits'][0],None,None,'task')
        selection=actual['pred_logits'][0].sigmoid().flatten().topk(3,sorted=False).indices//2
        # Merged output: parent object query 2, then part queries 2 and 3,
        # both of which were generated from parent object query 0.
        torch.testing.assert_close(result.parent_query,torch.tensor([2,0,0])[selection])
        torch.testing.assert_close(result.original,actual['pred_logits'][0])


if __name__=='__main__':
    unittest.main()
