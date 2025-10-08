from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Type, Literal, Any
from pydantic import BaseModel, Field, field_validator,PrivateAttr
import os
import sys
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path
import io, contextlib

import torch
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

# --- keep MaskDINO external ---
ROOT= ...
maskdino_repo_path = f"{ROOT}MaskDINO"  # set this to your MaskDINO repo path
if maskdino_repo_path not in sys.path:
    sys.path.append(maskdino_repo_path)
det_repo_path = f"{ROOT}detectron2"
if det_repo_path not in sys.path:
    sys.path.append(det_repo_path)


from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from maskdino.config import add_maskdino_config


class DINODetectionInput(BaseModel):
    """Input for dermatologic lesion detection tool (bounding boxes only)."""

    image_path: str = Field(
        ...,
        description="Path to the image (JPG/PNG/BMP). Panoramic clinical photos are supported."
    )
    box_format: Literal["xyxy", "xywh"] = Field(
        "xyxy",
        description="Bounding box format for the output list."
    )

    @field_validator("image_path")
    @classmethod
    def _check_exists(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise ValueError(f"File not found: {v}")
        return v


class DINODetectionTool(BaseTool):
    """
    Detects skin lesions in clinical images using Mask DINO (boxes only).
    Input: path to an image. Output: JSON with boxes/scores + a rendered image saved to disk.
    """

    name: str = "DINODetection"
    description: str = (
        "Detects dermatologic lesions in panoramic images using Mask DINO. The tool must be called only if the image is panoramic so if in the image are present more that one dermatological lesion"
        "Returns bounding boxes, scores, and the path to an image with boxes drawn. You don't need to give further deatails about the boxes, it is enough to show the image with the boxes drawn on it."
    )
    args_schema: Type[BaseModel] = DINODetectionInput

    # Runtime attributes
    device: Optional[str] = "cuda"
    _predictor: Any = PrivateAttr(default=None)  # Detectron2 DefaultPredictor
    _cfg: Any = PrivateAttr(default=None)
    class_name: str = "lesion"
    score_threshold: float = 0.5
    temp_dir: str = "temp"
    def __init__(
        self,
        *,
        config_path: str = "MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        weights_path: str = "./output/best_model.pth",#path to the weights of the model
        device: Optional[str] = "cuda",
        score_threshold : float = 0.5,#threshold for filtering weak detections
        temp_dir: Optional[str] = "temp"
    ):
        super().__init__()

        self.device = torch.device(device) if device else torch.device("cpu")

        # build cfg
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = "cuda" if (self.device.type == "cuda" and torch.cuda.is_available()) else "cpu"

        # leave test-time threshold low; per-call threshold will be applied manually
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(0.0)

        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        cfg.OUTPUT_DIR = os.path.dirname(os.path.abspath(weights_path))
        # sanity clamp score_threshold to [0,1]
        self.score_threshold = float(min(max(score_threshold, 0.0), 1.0))
        self._cfg = cfg
        with contextlib.redirect_stdout(io.StringIO()):
            self._predictor = DefaultPredictor(cfg)
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    # ---------- small helpers ----------
    def _pil_to_bgr_ndarray(self, img: Image.Image) -> np.ndarray:
        """Detectron2 expects BGR uint8 HxWx3."""
        arr = np.array(img.convert("RGB"), dtype=np.uint8)  # RGB
        return arr[:, :, ::-1].copy()  # to BGR
    def _boxes_xyxy_to_xywh(self, boxes_xyxy: List[List[float]]) -> List[List[float]]:
        out = []
        for x1, y1, x2, y2 in boxes_xyxy:
            out.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        return out
    def _draw_boxes(self, pil_img: Image.Image, boxes_xyxy: List[List[float]], scores: List[float]) -> Image.Image:
        draw = ImageDraw.Draw(pil_img)
        for (x1, y1, x2, y2), s in zip(boxes_xyxy, scores):
            rect = [int(x1), int(y1), int(x2), int(y2)]
            # rectangle (outline width ~2 px)
            draw.rectangle(rect, outline=(0, 255, 0), width=2)
            # simple score label
            label = f"{self.class_name}: {s:.2f}"
            text_bg = [rect[0], max(0, rect[1] - 16), rect[0] + 110, rect[1]]
            draw.rectangle(text_bg, fill=(0, 255, 0))
            draw.text((rect[0] + 2, max(0, rect[1] - 14)), label, fill=(0, 0, 0))
        return pil_img

    # ---------- core run ----------
    def _run(
        self,
        image_path: str,
        box_format: str = "xyxy",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Returns:
            results: {
               "detections": {
                   "boxes": [[...], ...],   # in requested box_format
                   "scores": [..],
                   "class_names": ["lesion", ...],
               },
               "boxed_image_path": "/path/to/image__dets.png"
            }
            metadata: {...}
        """
        try:
            

            # read & convert
            pil_img = Image.open(image_path).convert("RGB")
            bgr_img = self._pil_to_bgr_ndarray(pil_img)
            H, W = bgr_img.shape[:2]

            outputs = self._predictor(bgr_img)
            inst = outputs["instances"].to("cpu")

            # raw tensors
            boxes_xyxy_all = inst.pred_boxes.tensor.numpy().tolist() if inst.has("pred_boxes") else []
            scores_all = inst.scores.numpy().tolist() if inst.has("scores") else []

            # threshold filter (no cfg mutation)
            keep_idx = [i for i, s in enumerate(scores_all) if s >= self.score_threshold]
            boxes_xyxy = [boxes_xyxy_all[i] for i in keep_idx]
            scores = [scores_all[i] for i in keep_idx]

            # draw on a copy
            drawn = pil_img.copy()
            drawn = self._draw_boxes(drawn, boxes_xyxy, scores)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_img_path = (self.temp_dir / f"generated_detection_{Path(image_path).stem}_{timestamp}.png")
            drawn.save(out_img_path)

            # format for output
            if box_format == "xywh":
                boxes = self._boxes_xyxy_to_xywh(boxes_xyxy)
            else:
                boxes = [[float(a), float(b), float(c), float(d)] for a, b, c, d in boxes_xyxy]
            
            detections = {
                "boxes": boxes,
                "scores": [float(s) for s in scores],
                "class_names": [self.class_name for _ in scores],
            }

            metadata = {
                "image_path": image_path,
                "boxed_image_path": str(out_img_path),
                "image_size": {"width": int(W), "height": int(H)},
                "box_format": box_format,
                "score_threshold_applied": self.score_threshold,
                "model": "MaskDINO",
                "device": str(self.device),
                "note": "Scores in [0,1]. 'class_names' uses a single foreground class: 'lesion'.",
            }

            return {"detections": detections, "image_path": str(out_img_path)}, metadata

        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        box_format: str = "xyxy",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict, Dict]:
        return self._run(
            image_path=image_path,
            box_format=box_format,
            run_manager=run_manager,
        )