"""
RetinaSight — PyTorch to MATLAB Weight Bridge
SIH 2026, PS ID 26038, Team OnFocus

Exports the trained ResNet-50 PyTorch checkpoint (retinasight_resnet50.pth)
into a structured MATLAB .mat file (matlab/retinasight_resnet50_weights.mat)
for native Deep Learning Toolbox inference and Grad-CAM in MATLAB R2026a.
"""

from pathlib import Path
import scipy.io as sio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PTH = PROJECT_ROOT / "retinasight_resnet50.pth"
OUTPUT_MAT = PROJECT_ROOT / "matlab" / "retinasight_resnet50_weights.mat"


def export_weights():
    print(f"[EXPORT] Loading PyTorch checkpoint: {MODEL_PTH}")
    state = torch.load(str(MODEL_PTH), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    mat_dict = {}
    for key, tensor in state.items():
        arr = tensor.numpy()
        # Clean key for MATLAB variable naming rules
        safe_key = key.replace(".", "_")
        mat_dict[safe_key] = arr

    # Explicitly store FC classification head weights & bias
    if "fc.weight" in state:
        mat_dict["fc_weights"] = state["fc.weight"].numpy()  # [5, 2048]
        mat_dict["fc_bias"] = state["fc.bias"].numpy().reshape(-1, 1)  # [5, 1]
    elif "fc.5.weight" in state:
        mat_dict["fc_weights"] = state["fc.5.weight"].numpy()
        mat_dict["fc_bias"] = state["fc.5.bias"].numpy().reshape(-1, 1)

    sio.savemat(str(OUTPUT_MAT), mat_dict)
    print(f"[SUCCESS] Exported {len(mat_dict)} tensors to: {OUTPUT_MAT}")


if __name__ == "__main__":
    export_weights()
