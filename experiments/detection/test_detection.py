
"""
Static smoke test for DINODetectionTool.
All paths and parameters are set as static variables below.
"""

import json
import os
from pathlib import Path
import warnings
from DermAgent.tools.detection import DINODetectionTool
# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# --------------------------
# STATIC VARIABLES
# --------------------------
ROOTP="/home/jovyan/python/"
IMAGE_PATH   = f"{ROOTP}MedMe/datasets/Panoramic/nevi-atipici-2a.jpg"
WEIGHTS_PATH = f"/home/jovyan/nfs/lsgroi/output/best_model.pth"
CONFIG_PATH  = f"{ROOTP}MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
DEVICE       = "cuda"   # or "cpu"
SCORE_TH     = 0.4
BOX_FORMAT   = "xyxy"   # or "xywh"
NUM_CLASSES  = 1        # lesion class only
# --------------------------

def main():
    # sanity checks
    for p in [IMAGE_PATH, WEIGHTS_PATH, CONFIG_PATH]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing file: {p}")

    # init tool
    tool = DINODetectionTool(
        config_path=CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        device=DEVICE,
        score_threshold=SCORE_TH
    )

    # run inference
    results, meta = tool._run(
        image_path=IMAGE_PATH,
        box_format=BOX_FORMAT,
    )

    print("\n=== Detections ===")
    print(json.dumps(results.get("detections", {}), indent=2))

    boxed_path = results.get("boxed_image_path") or meta.get("boxed_image_path")
    print(f"\nSaved boxed image to: {boxed_path}")

    if not boxed_path or not Path(boxed_path).is_file():
        raise RuntimeError("boxed_image_path was not created.")

    print("\nSmoke test OK ✅")

if __name__ == "__main__":
    main()