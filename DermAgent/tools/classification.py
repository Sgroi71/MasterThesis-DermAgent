from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field
import torch
import torchvision
from torch import nn
from DermAgent.tools.SwinModelMultiTask import SwinModelMultiTask
from PIL import Image

from DermAgent.tools.utils import transform_factory, load_checkpoint

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool



class MuteClassificationInput(BaseModel):
    """Input for dermatologic image analysis tools. Only supports JPG, PNG, bmp  images."""

    image_path: str = Field(
        ..., description="Path to the dermatologic image file, only supports JPG, PNG, bmp images"
    )

class MuteClassifierTool(BaseTool):
    """A tool that analyzes dermatologic images and classifies them for multiple pathologies using SwinModelMultiTask."""

    name: str = "Mute_classifier"
    description: str = (
        "A tool that analyzes dermatologic images and classifies them for multiple pathologies. "
        "Input should be the path to a dermatologic image file "
        "Output is a dictionary with heads (of the model, directly connected to dataset used to train them) as key and a dictionary of pathologies and their predicted probabilities (0 to 1) as values." \
        "Example: {'fitzpatrick': {'acanthosis nigricans':0.9987\}\}" \
        "Reason over all the output heads of the model in order to output only one pathology among all of them"
    )
    args_schema: Type[BaseModel] = MuteClassificationInput
    model: SwinModelMultiTask = None
    device: Optional[str] = "cuda"
    transform: torchvision.transforms.Compose = None
    head: str = "All"

    def __init__(self, device: Optional[str] = "cuda", config_path: str = "./checkpoints/exp-HAM+Derm7pt-all+BCN+HAM-bin+DermNet+Fitzpatrick.yaml",output_head: str = "All"):
        super().__init__()
        self.model = load_checkpoint(config_path)
        self.device = torch.device(device) if device else "cpu"
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transform_factory("classification", "test") 
        self.head= output_head

    def _process_image(self, image_file_name):
        # load image and transform into tensor....
        image = Image.open(image_file_name).convert('RGB')  # Open and convert to RGB
        original_size = image.size  # Store original image shape
        inputs = self.transform(image)
        inputs = inputs.unsqueeze(0).to(self.device)
        return inputs

    def _run(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify dermatologic images for multiple pathologies.

        Args:
            image_path (str): The path to the image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[Dict[str,Dict[str, float]], Dict]: A tuple containing the classification results
                                           (heads as keys and a dictionary as value with pathologies as keys and their probabilities from 0 to 1 as value)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        try:
            img = self._process_image(image_path)

            with torch.inference_mode():
                outputs = self.model.forward_classification_tasks(img)
            total_outputs={}
            if self.head != "All":
                outputs = outputs[self.head]['predictions'].tolist()
                # Convert to dictionary with pathology names as keys and probabilities as values as keys contained in the dataset_info (attribute of the model): model.dataset_info[outlabel]['class_labels']
                pathology_labels = self.model.dataset_info[self.head]['class_labels']
                outputs = dict(zip(pathology_labels, outputs))
                total_outputs[self.head]=outputs
            else:
                for h in outputs:
                    pred = outputs[h]['predictions'].tolist()
                    # Convert to dictionary with pathology names as keys and probabilities as values as keys contained in the dataset_info (attribute of the model): model.dataset_info[outlabel]['class_labels']
                    pathology_labels = self.model.dataset_info[h]['class_labels']
                    o = dict(zip(pathology_labels, pred))
                    total_outputs[h]=o
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }
            return total_outputs, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        #head: str = "Fitzpatrick",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify dermatologic image for multiple pathologies.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            image_path (str): The path to the image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(image_path, run_manager)