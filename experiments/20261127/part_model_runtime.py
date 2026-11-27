"""Isolated adapters for original PartGLEE/VLPart code and public checkpoints.

No project config.activate(): yolo-lib contains minimal dependency shims which
must not shadow real scipy/requests installed in the separate parts environment.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
PARTGLEE=ROOT/"experiments/vendor/PartGLEE-master"
VLPART=ROOT/"experiments/vendor/VLPart-main"
CACHE=ROOT/"experiments/model_cache/parts"


def prepare_port():
    """Enable the author's existing grid_sample fallback without CUDA build."""
    file=PARTGLEE/"projects/PartGLEE/partglee/models/pixel_decoder/ops/functions/ms_deform_attn_func.py"
    text=file.read_text(encoding="utf-8")
    old="    raise ModuleNotFoundError(info_string)"
    new="    MSDA = None  # Local inference port: caller already falls back to author's PyTorch core."
    if old in text:
        before=hashlib.sha256(file.read_bytes()).hexdigest()
        file.write_text(text.replace(old,new),encoding="utf-8")
        change=dict(file=str(file.relative_to(ROOT)),before_sha256=before,
                    after_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    reason="Allow module import without compiled MSDA; original caller already catches failure and uses its PyTorch reference implementation.")
        (ROOT/"experiments/20261127/part_model_port.json").write_text(json.dumps(change,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Unused training-only import removed upstream from newer Detectron2.
    file=VLPART/"vlpart/modeling/roi_heads/fast_rcnn_vlm.py"
    text=file.read_text(encoding="utf-8")
    line="from detectron2.data.detection_utils import get_fed_loss_cls_weights\n"
    if line in text:
        file.write_text(text.replace(line,""),encoding="utf-8")
    file=VLPART/"vlpart/modeling/meta_arch/vlm_rcnn.py"
    text=file.read_text(encoding="utf-8")
    line="            padding_constraints=self.backbone.padding_constraints,\n"
    if line in text:
        # Standard timm FPN has only size_divisibility, no extra constraints.
        file.write_text(text.replace(line,""),encoding="utf-8")


def configure_paths(model):
    for path in (PARTGLEE,PARTGLEE/"projects/PartGLEE",ROOT/"experiments/vendor/CLIP-main",VLPART):
        if path.exists():
            sys.path.insert(0,str(path))
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/parts-mpl"))
    os.environ.setdefault("HF_HOME",str(ROOT/"experiments/model_cache/huggingface"))
    from PIL import Image
    if not hasattr(Image,"LINEAR"):
        Image.LINEAR=Image.Resampling.BILINEAR
    import detectron2.layers as layers
    if not hasattr(layers,"move_device_like"):
        layers.move_device_like=lambda source,target:source.to(target.device)
    os.chdir(PARTGLEE if model=="partglee" else VLPART)


def load_model(model_name):
    import torch
    prepare_port()
    configure_paths(model_name)
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    cfg=get_cfg()
    if model_name=="vlpart":
        # The vendored Detectron2 registers LVIS names also registered by VLPart.
        # No benchmark dataset is loaded in this image-only experiment.
        from detectron2.data import DatasetCatalog,MetadataCatalog
        DatasetCatalog.clear();MetadataCatalog.clear()
        from vlpart.config import add_vlpart_config
        add_vlpart_config(cfg)
        cfg.merge_from_file(str(VLPART/"configs/pascal_part/r50_pascalpart.yaml"))
        cfg.MODEL.TIMM.PRETRAINED=False
        cfg.MODEL.WEIGHTS=str(CACHE/"r50_pascalpart.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=.5
        from vlpart.data.datasets.pascal_part import PASCAL_PART_CATEGORIES
        labels=[r["name"] for r in PASCAL_PART_CATEGORIES]
    elif model_name=="partglee":
        from detectron2.projects.partglee import add_partglee_config
        add_partglee_config(cfg)
        cfg.merge_from_file(str(PARTGLEE/"projects/PartGLEE/configs/Inference/RN50/pascal_part.yaml"))
        cfg.MODEL.WEIGHTS=str(CACHE/"PartGLEE_RN50.pth")
        clip_file=PARTGLEE/"projects/PartGLEE/clip_vit_base_patch32/pytorch_model.bin"
        if not clip_file.exists():
            state=torch.load(cfg.MODEL.WEIGHTS,map_location="cpu",weights_only=False)
            prefix="partglee.text_encoder."
            text_state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            if len(text_state)<100:
                raise RuntimeError("Checkpoint does not contain the CLIP text encoder")
            torch.save(text_state,clip_file)
            del state,text_state
        labels=["bicycle","bicycle wheel"]
    else:
        raise ValueError(model_name)
    cfg.MODEL.DEVICE="cuda"
    cfg.MODEL.RESNETS.NORM="FrozenBN" if model_name=="partglee" else cfg.MODEL.RESNETS.NORM
    cfg.freeze()
    model=build_model(cfg).eval()
    state=torch.load(cfg.MODEL.WEIGHTS,map_location="cpu",weights_only=False)
    state=state.get("model",state)
    incompatible=model.load_state_dict(state,strict=False)
    missing_parameters=set(incompatible.missing_keys)&set(dict(model.named_parameters()))
    if missing_parameters:
        raise RuntimeError("Missing model parameters: "+str(sorted(missing_parameters)[:15]))
    info=dict(missing_keys=incompatible.missing_keys,unexpected_keys=incompatible.unexpected_keys,
              missing_parameter_count=len(missing_parameters))
    return model,cfg,labels,info


if __name__=="__main__":
    import argparse,traceback
    parser=argparse.ArgumentParser();parser.add_argument("model",choices=["vlpart","partglee"])
    args=parser.parse_args()
    try:
        model,cfg,labels,extra=load_model(args.model)
        print("MODEL_READY",args.model,len(labels),sum(p.numel() for p in model.parameters()),flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
