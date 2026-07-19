"""One-off script: exports AlignCNN + AlignHead's trained weights from
jepa_heads_synthetic.pt into a plain JSON file a hand-written JS forward
pass can consume, for the in-browser real-time JEPA demo. Not part of the
Flask app - run manually when the checkpoint changes."""
import json
import torch

ckpt = torch.load("jepa_heads_synthetic.pt", map_location="cpu", weights_only=False)


DECIMALS = 4  # cuts JSON size vs full float64 repr; negligible accuracy impact for a demo


def _round(nested):
    if isinstance(nested, list):
        return [_round(x) for x in nested]
    return round(nested, DECIMALS)


def conv_block(prefix, state):
    return {
        "conv_w": _round(state[f"{prefix}.0.weight"].tolist()),
        "conv_b": _round(state[f"{prefix}.0.bias"].tolist()),
        "bn_w": _round(state[f"{prefix}.1.weight"].tolist()),
        "bn_b": _round(state[f"{prefix}.1.bias"].tolist()),
        "bn_mean": _round(state[f"{prefix}.1.running_mean"].tolist()),
        "bn_var": _round(state[f"{prefix}.1.running_var"].tolist()),
    }


def linear(prefix, state):
    return {"w": _round(state[f"{prefix}.weight"].tolist()), "b": _round(state[f"{prefix}.bias"].tolist())}


cnn_state = ckpt["align_cnn_state"]
head_state = ckpt["align_head_state"]

out = {
    "meta": {
        "win_size": ckpt["win_size"],
        "max_offset_px": ckpt["max_offset_px"],
        "align_metrics": ckpt["align_metrics"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    },
    "trunk": [conv_block(f"trunk.{i}", cnn_state) for i in range(4)],
    "rot_net": [linear("rot_net.0", head_state), linear("rot_net.3", head_state), linear("rot_net.5", head_state)],
    "offset_net": [linear("offset_net.0", head_state), linear("offset_net.3", head_state), linear("offset_net.5", head_state)],
}

with open("align_weights.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

import os
print("wrote align_weights.json,", os.path.getsize("align_weights.json"), "bytes")
