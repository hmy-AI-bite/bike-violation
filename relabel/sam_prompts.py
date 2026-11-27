"""Automatic weak instance guidance, independent of human contact annotations."""
import numpy as np


def center_point_prompt(box, width, height):
    """SAM foreground point at box center in original processed-image pixels.

    This is not a ground-contact estimate. In occluded scenes the center may
    belong to another object, so callers must explicitly opt into this prompt.
    """
    coords=np.asarray(box,dtype=float)
    if coords.shape!=(4,) or not np.isfinite(coords).all():
        raise ValueError("box must contain four finite pixel coordinates")
    if width<1 or height<1 or coords[2]<=coords[0] or coords[3]<=coords[1]:
        raise ValueError("image dimensions and box area must be positive")
    center=(coords[:2]+coords[2:])/2
    center=np.clip(center,[0,0],[width-1,height-1]).astype(np.float32)
    return dict(point_coords=center[None,:],point_labels=np.ones(1,dtype=np.int32))
