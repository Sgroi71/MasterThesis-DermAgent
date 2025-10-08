from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import tempfile
import torch
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import traceback

from datetime import datetime
from PIL import Image
import captum
from captum.attr import Occlusion, Saliency, InputXGradient,NoiseTunnel,IntegratedGradients,KernelShap
from captum.attr import visualization as viz
from skimage.segmentation import slic
from skimage.util import img_as_float

from DermAgent.agent import *
from DermAgent.tools import *
from DermAgent.utils import *


class ExplanationInput(BaseModel):
    """Input schema for the Explanation Tool."""

    # Required fields
    
    input_img_path: str = Field(
        ...,
        description="The local path of the image to be explained."
    )
    targetClass: str = Field(
        ...,
        description="The target class for the image. This value must be selected among the classes of the selected head"
    )
    head: str = Field(
        ...,
        description="The head of the classification model used for prediction. This value must be selected among the following [HAM10k,DERM7pt-derm,DERM7pt-clinic,BCN20k,HAM10k-bin,Fitzpatrick,DermNet]"
    )
    


class ExplanationTool(BaseTool):
    """Tool for generating a silency map for a given image in order to explain the Mute classifier model's prediction."""

    name: str = "explanation_tool"
    description: str = (
        "A tool that generates a saliency map for a given image to explain the classification model's prediction. Before calling this tool, you must have already classified the image using the MuteClassifierTool. Pay attention to use the same head used for the classification. And pay attention to the spelling of the target class, it must be exactly the same as in the model's classes. "
        "It takes an image path, and return the saliency maps, the generated pictures represent in the left the original image in the center the image masking and in the right the blended heatmap "
        "that highlights the regions of the image that contributed most to the prediction."
    )
    args_schema: Type[BaseModel] = ExplanationInput
    transform: torchvision.transforms.Compose = None
    norm_transform:torchvision.transforms.Compose = None
    model: SwinModelMultiTask = None
    device: Optional[str] = "cuda"
    explanation_technique: str = "vanilla_gradient"
    head: str = "Fitzpatrick"
    temp_dir: Optional[str] = "temp"
    def __init__(
        self,
        device: Optional[str] = "cuda",
        config_path: str = "./checkpoints/exp-HAM+Derm7pt-all+BCN+HAM-bin+DermNet+Fitzpatrick.yaml",
        temp_dir: Optional[str] = "temp",
    ):
        super().__init__()
        self.device = torch.device(device) if device else "cpu"
        self.model = load_checkpoint(config_path).to(self.device)
        self.model.eval()
        self.transform, self.norm_transform = transform_factory("explanation", "test")
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)
        
        

    def _process_image(self, image_file_name):
        """
        Process the input image for the model.
        Args:
            image_file_name (str): Path to the input image file.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray]: Transformed image tensor, input image tensor for the model, and original image as a numpy array.
        """
        # load image and transform into tensor....
        image = Image.open(image_file_name).convert('RGB')  # Open and convert to RGB
        transformed_img = self.transform(image)
        input_img = self.norm_transform(transformed_img).unsqueeze(0).to(self.device)
        orig_img = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0))
        transformed_img=transformed_img.unsqueeze(0).to(self.device)
        input_img.requires_grad_()
        return transformed_img, input_img, orig_img
    
    def forward_model(self,image: torch.Tensor):
        """ Forward function for the model used in captum
        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model predictions.
        """

        outputs = self.model.forward_explanation_tasks(image)
        predictions = outputs[self.head]['predictions']
        return predictions

    def generate_feature_mask(self, input_img):
        """ Generate a feature mask for SHAP using SLIC superpixels.
        Args:
            input_img (np.ndarray): Input image as a numpy array.
        Returns:
            torch.Tensor: Feature mask tensor.
        """
        # Step 1: Converti orig_img in float [0, 1]
        orig_img_float = img_as_float(input_img)  # shape (H, W, 3), values in [0, 1]

        # Step 2: Crea superpixel con SLIC
        # n_segments: numero di superpixel (aumenta per maggiore granularità)
        segments = slic(orig_img_float, n_segments=100, compactness=10)

        # Step 3: Converti in torch tensor e manda su device
        feature_mask = torch.tensor(segments, dtype=torch.long).to(self.device)
        return feature_mask

    def generateAttribution_Vanilla(self, input_img, pred_label_idx,SG=False, stdevs=0.3, nt_samples=30, nt_samples_batch_size=5):
        """
        Generate vanilla gradient attributions.
        Args:
            input_img (torch.Tensor): Input image tensor.
            pred_label_idx (torch.Tensor): Target class index tensor.
            SG (bool): Whether to use SmoothGrad.
            stdevs (float): Standard deviation for noise in SmoothGrad.
            nt_samples (int): Number of samples for Noise Tunnel.
            nt_samples_batch_size (int): Batch size for Noise Tunnel samples.
        Returns:
            torch.Tensor: Attribution tensor.
        """
        vanilla_gradient = Saliency(self.forward_model)
        if not SG:
            return vanilla_gradient.attribute(input_img, target=pred_label_idx)
        else:
            smoothGrad = NoiseTunnel(vanilla_gradient)
            return smoothGrad.attribute(
                        input_img, 
                        stdevs=stdevs, 
                        target=pred_label_idx,
                        nt_samples=nt_samples, 
                        nt_samples_batch_size=nt_samples_batch_size)

    def generateAttribution_IxG(self, input_img, pred_label_idx, SG=False, stdevs=0.3, nt_samples=30, nt_samples_batch_size=5):
        """ Generate Input x Gradient attributions.
        Args:
            input_img (torch.Tensor): Input image tensor.
            pred_label_idx (torch.Tensor): Target class index tensor.
            SG (bool): Whether to use SmoothGrad.
            stdevs (float): Standard deviation for noise in SmoothGrad.
            nt_samples (int): Number of samples for Noise Tunnel.
            nt_samples_batch_size (int): Batch size for Noise Tunnel samples.
        Returns:
            torch.Tensor: Attribution tensor.
        """
        
        InXgrad = InputXGradient(self.forward_model)
        if not SG:
            return InXgrad.attribute(input_img, target=pred_label_idx)
        else:
            smoothGrad = NoiseTunnel(InXgrad)
            return smoothGrad.attribute(
                        input_img, 
                        stdevs=stdevs, 
                        target=pred_label_idx,
                        nt_samples=nt_samples, 
                        nt_samples_batch_size=nt_samples_batch_size)
    def generateAttribution_IntegratedGradient(self, input_img, pred_label_idx, n_steps=200, internal_batch_size=10, SG=False, stdevs=0.3, nt_samples=30, nt_samples_batch_size=5):
        """ Generate Integrated Gradient attributions.
        Args:
            input_img (torch.Tensor): Input image tensor.
            pred_label_idx (torch.Tensor): Target class index tensor.
            n_steps (int): Number of steps for Integrated Gradients.
            internal_batch_size (int): Batch size for internal computations.
            SG (bool): Whether to use SmoothGrad.
            stdevs (float): Standard deviation for noise in SmoothGrad.
            nt_samples (int): Number of samples for Noise Tunnel.
            nt_samples_batch_size (int): Batch size for Noise Tunnel samples.
        Returns:
            torch.Tensor: Attribution tensor.
        """ 
        ig = IntegratedGradients(self.forward_model)
        if not SG:
            return ig.attribute(
                                input_img, 
                                target=pred_label_idx, 
                                n_steps=n_steps,
                                internal_batch_size=internal_batch_size)
        else:
            smoothGrad_ig= NoiseTunnel(ig)
            return smoothGrad_ig.attribute(
                                input_img, 
                                stdevs=stdevs, 
                                target=pred_label_idx,
                                nt_samples=nt_samples, 
                                nt_samples_batch_size=nt_samples_batch_size,
                                n_steps=n_steps,
                                internal_batch_size=internal_batch_size)

    def generateAttribution_Occlusion(self, input_img, pred_label_idx, sliding_window_shapes=(3, 15, 15), strides=(3, 8, 8)):
        """ Generate Occlusion attributions.
        Args:
            input_img (torch.Tensor): Input image tensor.
            pred_label_idx (torch.Tensor): Target class index tensor.
            sliding_window_shapes (Tuple[int, int, int]): Shapes of the sliding window.
            strides (Tuple[int, int, int]): Strides of the sliding window.
        Returns:
            torch.Tensor: Attribution tensor.
        """
        occlusion = Occlusion(self.forward_model)
        attributions_occ = occlusion.attribute(input_img, target=pred_label_idx, sliding_window_shapes=sliding_window_shapes, strides=strides)
        return attributions_occ

    def generateAttribution_SHAP(self, input_img, orig_img,pred_label_idx, n_samples=400, perturbations_per_eval=32,show_progress=False):
        """ Generate SHAP attributions.
        Args:
            input_img (torch.Tensor): Input image tensor.
            orig_img (torch.Tensor): Original image tensor.
            pred_label_idx (torch.Tensor): Target class index tensor.
            n_samples (int): Number of samples for SHAP.
            perturbations_per_eval (int): Number of perturbations per evaluation.
            show_progress (bool): Whether to show progress.
        Returns:
            torch.Tensor: Attribution tensor.
        """
        feature_mask = self.generate_feature_mask(orig_img)

        # Shift the feature mask so that it starts at 0
        feature_mask = feature_mask - feature_mask.min()

        kernel_shap = KernelShap(self.forward_model)

        attributions_ks = kernel_shap.attribute(input_img.to(self.device),
                                        target=pred_label_idx,
                                        n_samples=n_samples,
                                        perturbations_per_eval=perturbations_per_eval,
                                        show_progress=show_progress,
                                        feature_mask=feature_mask)
        return attributions_ks


    def create_saliency_map(self, attributions, orig_img, alpha_overlay=0.5):
        """ Create a saliency map visualization.
        Args:
            attributions (torch.Tensor): Attribution tensor.
            orig_img (np.ndarray): Original image as a numpy array.
            alpha_overlay (float): Alpha value for the overlay.
        Returns:
            np.ndarray: Saliency map visualization.
        """
        return viz.visualize_image_attr_multiple(
                                        np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        orig_img,
                                        methods=["original_image", "blended_heat_map", "masked_image"],
                                        signs=["all", "positive", "positive"],
                                        alpha_overlay=alpha_overlay,
                                        show_colorbar=True,
                                        titles=["Original", "Positive Attribution", "Masked"],
                                        use_pyplot=False,
                                    )
        
    def attribution_to_mask(self, attributions, percentile=90):
        """ Convert attributions to a binary mask based on a percentile threshold.
        Args:
            attributions (torch.Tensor): Attribution tensor.
            percentile (float): Percentile threshold for mask generation.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Binary mask and normalized attribution map.
        """
        attr = attributions.squeeze().detach().cpu().numpy()  # (C,H,W) o (H,W)
        attr = np.clip(attr, a_min=0, a_max=None)
        if attr.ndim == 3:
            attr = attr.mean(axis=0)  # (H,W) – puoi usare sum o L2 se preferisci
        elif attr.ndim != 2:
            raise ValueError(f"Unexpected attribution ndim={attr.ndim}. Expected 2 or 3.")
        amax = attr.max()
        if amax > 0:
            attr = attr / amax
        thr = np.percentile(attr, percentile)
        mask = (attr >= thr).astype(np.uint8)   # (H,W) {0,1}
        return mask, attr

    def iou(self, mask_a, mask_b):
        """ Compute Intersection over Union (IoU) between two binary masks.
        Args:
            mask_a (np.ndarray): First binary mask.
            mask_b (np.ndarray): Second binary mask.
        Returns:
            float: IoU value.
        """
        mask_a = self._ensure_binary_uint8(mask_a)
        mask_b = self._ensure_binary_uint8(mask_b)
        # stesse dimensioni
        mask_a, mask_b = self._match_shapes(mask_a, mask_b)
        inter = (mask_a & mask_b).sum()
        union = (mask_a | mask_b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def PlotIoUMaskOverPercentile(self, attributions, gt_mask):
        """ Plot IoU vs Percentile for attribution masks against a ground truth mask.
        Args:
            attributions (torch.Tensor): Attribution tensor.
            gt_mask (np.ndarray): Ground truth binary mask.
        Returns:
            matplotlib.figure.Figure: Plot figure.
        """
        percentiles = list(range(50, 100, 5))
        ious = []
        for p in percentiles:
            pred_mask, _ = self.attribution_to_mask(attributions, percentile=p)  # (H,W)
            # allinea qui, così l’IoU non va in errore
            _, gt_aligned = self._match_shapes(pred_mask, gt_mask)
            val_iou = self.iou(pred_mask, gt_aligned)
            ious.append(val_iou)

        plt.figure()
        plt.plot(percentiles, ious, marker='o')
        plt.xlabel("Percentile")
        plt.ylabel("IoU")
        plt.title("IoU medio vs Percentile")
        return plt
    
    def visualize_masks(self, attributions, seg_mask, percentiles=[70,80,90]):
        """ Visualize attribution masks at different percentiles alongside the ground truth segmentation mask.
        Args:
            attributions (torch.Tensor): Attribution tensor.
            seg_mask (np.ndarray): Ground truth binary segmentation mask.
            percentiles (List[int]): List of percentiles to visualize.
        Returns:
            matplotlib.figure.Figure: Visualization figure.
        """
        fig, axs = plt.subplots(1, len(percentiles)+2, figsize=(15,5))

        # Heatmap 2D normalizzata per visualizzazione
        _, attr2d = self.attribution_to_mask(attributions, percentile=90)
        axs[0].imshow(attr2d, cmap="inferno"); axs[0].set_title("Attributions (norm)"); axs[0].axis("off")

        # Allinea la GT all’attr prima di mostrare
        seg_mask = self._ensure_binary_uint8(seg_mask)
        seg_mask = self._resize_mask_to(seg_mask, attr2d.shape)
        axs[1].imshow(seg_mask, cmap="gray"); axs[1].set_title("GT Segmentation"); axs[1].axis("off")

        # Maschere ai diversi percentili (già (H,W) come attr2d)
        for i, p in enumerate(percentiles):
            mask_p, _ = self.attribution_to_mask(attributions, percentile=p)
            axs[i+2].imshow(mask_p, cmap="gray")
            axs[i+2].set_title(f"Percentile {p}")
            axs[i+2].axis("off")

        plt.tight_layout()
        return plt
    def load_segmentation_mask(self,path):
        """
        load a segmentation mask from a given path and convert it to a binary numpy array.
        Args:
            path (str): Path to the segmentation mask image file.
        Returns:
            np.ndarray: Binary segmentation mask as a numpy array.
        """
        img = Image.open(path).convert("L")  # grayscale
        arr = np.array(img)

        # Binarizza: valori >127 = 1
        mask = (arr > 127).astype(np.uint8)
        return mask
    
    def _ensure_binary_uint8(self, m):
        m = (m > 0).astype(np.uint8)
        return m

    def _resize_mask_to(self, mask_hw, target_hw):
        """mask_hw: (H,W) array {0,1}; target_hw: (H,W) -> return (H,W) uint8"""
        Ht, Wt = target_hw
        pil = Image.fromarray((mask_hw * 255).astype(np.uint8))
        pil = pil.resize((Wt, Ht), resample=Image.NEAREST)
        out = (np.array(pil) > 127).astype(np.uint8)
        return out

    def _match_shapes(self, a, b):
        """
        Return (a2, b2) with the same shape.
        If different: resize b to a.
        """
        if a.shape != b.shape:
            b = self._resize_mask_to(b, a.shape)
        return a, b
    


    def _run(
        self,
        input_img_path: str,
        targetClass: str,
        head: str,
        explanation_technique: Optional[str] = "vanilla_gradient",
        n_steps: int = 200,
        sliding_window_shapes: Tuple[int, int, int] = (3, 15, 15),
        strides: Tuple[int, int, int] = (3, 8, 8),
        internal_batch_size: int = 10,
        n_samples: int = 400,
        perturbations_per_eval: int = 32,
        SG: bool = True,
        stdevs: float = 0.2,
        nt_samples: int = 30,
        nt_samples_batch_size: int = 5,
        alpha_overlay: float = 0.5,
        output_image_path: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """ Generate a silency map for a given image in order to explain the Mute model's prediction.
        Args:
            explanation_technique: The explanation technique to use. Options are: integrated_gradients, occlusion, input_x_gradient, vanilla_gradient, shap.
            input_img_path: The local path of the image to be explained.
            targetClass: The target class for the image.
            head: The head of the model used for prediction.
            prediction: Optional predicted label for the image. Only used for visualization purposes.
            gt: Optional ground truth label for the image. Only used for visualization purposes.
            n_steps: Number of steps for integrated gradients.
            sliding_window_shapes: Shapes of the sliding window for occlusion.
            strides: Strides of the sliding window for occlusion.
            internal_batch_size: Batch size for internal computations for integrated gradients.
            n_samples: Number of samples for SHAP.
            perturbations_per_eval: Number of perturbations per evaluation for SHAP.
            SG: Whether to use smooth gradients.
            stdevs: Standard deviation for noise in smooth gradients.
            nt_samples: Number of samples for noise tunnel.
            nt_samples_batch_size: Batch size for noise tunnel samples.
            alpha_overlay: Alpha value for the overlay of the saliency map on the original image.
            output_image_path: Optional path to save the output image.
            run_manager: Optional callback manager for tool run.
        Returns:
            Tuple[Dict, Dict]: Output dictionary with output image path and metadata dictionary
        """

        
        try:
            self.explanation_technique = explanation_technique
            self.head = head

            # Load and process image
            transformed_img, input_img, orig_img = self._process_image(input_img_path)

            if transformed_img is None or not isinstance(transformed_img, torch.Tensor):
                raise ValueError("Failed to process the image. The input tensor is invalid.")
            #retrieve class index
            pathology_labels = self.model.dataset_info[self.head]['class_labels']
            # convert pathology_labels to lower case
            pathology_labels = [x.lower() for x in pathology_labels]
            
            
            if targetClass.lower() in pathology_labels:
                targetClass = targetClass.lower()
            else:
                raise ValueError(f"Target class {targetClass} not in pathology labels.")
            pred_label_idx = pathology_labels.index(targetClass.lower())
            
            pred_label_idx=torch.tensor(pred_label_idx).to(self.device)
            if self.explanation_technique == "integrated_gradients":
                attributions = self.generateAttribution_IntegratedGradient(
                    input_img,
                    pred_label_idx,
                    n_steps=n_steps,
                    SG=SG,
                    stdevs=stdevs,
                    nt_samples=nt_samples,
                    nt_samples_batch_size=nt_samples_batch_size,
                    internal_batch_size=internal_batch_size
                )
                if attributions is None:
                    raise ValueError("Failed to compute attributions. The result is None.")

            elif self.explanation_technique == "vanilla_gradient":
                attributions = self.generateAttribution_Vanilla(input_img, pred_label_idx,SG=SG,
                    stdevs=stdevs,
                    nt_samples=nt_samples,
                    nt_samples_batch_size=nt_samples_batch_size)
                if attributions is None:
                    raise ValueError("Failed to compute attributions. The result is None.")
            elif self.explanation_technique == "input_x_gradient":
                attributions = self.generateAttribution_IxG(
                    input_img,
                    pred_label_idx,
                    SG=SG,
                    stdevs=stdevs,
                    nt_samples=nt_samples,
                    nt_samples_batch_size=nt_samples_batch_size
                )
                if attributions is None:
                    raise ValueError("Failed to compute attributions. The result is None.")
            elif self.explanation_technique == "occlusion":
                attributions = self.generateAttribution_Occlusion(input_img, pred_label_idx, sliding_window_shapes=sliding_window_shapes, strides=strides)
                if attributions is None:
                    raise ValueError("Failed to compute attributions. The result is None.")
            elif self.explanation_technique == "shap":
                attributions = self.generateAttribution_SHAP(
                    input_img,
                    orig_img,
                    pred_label_idx,
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=False
                )
                if attributions is None:
                    raise ValueError("Failed to compute attributions. The result is None.")
            else:
                raise ValueError(f"explanation technique {self.explanation_technique} not supported. supported techniques are: integrated_gradients, occlusion, input_x_gradient, vanilla_gradient, shap.")

            fig, axis = self.create_saliency_map(attributions, orig_img, alpha_overlay=alpha_overlay)
            if output_image_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = (self.temp_dir / f"generated_explanation_{Path(input_img_path).stem}_{timestamp}_{self.explanation_technique}.png")
            else:
                outdir = Path(output_image_path)
                outdir.mkdir(parents=True, exist_ok=True)
                image_path = outdir / f"generated_explanation_{Path(input_img_path).stem}_{self.explanation_technique}_{targetClass.replace('/', '_')}.png"

            fig.savefig(str(image_path), format="png")
            plt.close(fig)

            output = {
                "image_path": str(image_path)
            }
            # Extract the x and y coordinates of the highest importance region
            mean_attribution = np.mean(attributions.squeeze().cpu().detach().numpy())
            std_dev_attribution = np.std(attributions.squeeze().cpu().detach().numpy())
            metadata = {
                "explanation_type": self.explanation_technique,
                "input_image_path": input_img_path,
                "target": targetClass,
                "aggregation_summary": {
                    "mean_attribution": mean_attribution,
                    "std_dev_attribution": std_dev_attribution
                }
            }

            return output, metadata

        except Exception as e:
            traceback.print_exc()  
            return (
                {"error": str(e)},
                {
                    "input_image_path": input_img_path,
                    "targetClass": targetClass,
                    "head": head,
                    "analysis_status": "failed",
                }
            )

    async def _arun(
        self,
        input_img_path: str,
        targetClass: str,
        head: str,
        explanation_technique: Optional[str] = "vanilla_gradient",
        n_steps: int = 200,
        sliding_window_shapes: Tuple[int, int, int] = (3, 15, 15),
        strides: Tuple[int, int, int] = (3, 8, 8),
        internal_batch_size: int = 10,
        n_samples: int = 400,
        perturbations_per_eval: int = 32,
        SG: bool = True,
        stdevs: float = 0.2,
        nt_samples: int = 30,
        nt_samples_batch_size: int = 5,
        alpha_overlay: float = 0.5,
        output_image_path: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Async version of _run."""
        return self._run(
            input_img_path=input_img_path,
            targetClass=targetClass,
            head=head,
            explanation_technique=explanation_technique,
            n_steps=n_steps,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            internal_batch_size=internal_batch_size,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            SG=SG,
            stdevs=stdevs,
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_samples_batch_size,
            alpha_overlay=alpha_overlay,
            output_image_path=output_image_path,
            run_manager=run_manager,
        )