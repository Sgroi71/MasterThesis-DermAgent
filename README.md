# Integrating Multi-Modal Reasoning and Explainable AI for Dermatological Image Analysis via LLM-Orchestrated Toolchains

**Author:** Leonardo Sgroi  
**Supervisors:** Flavio Giobergia, Ignazio Gallo  
**Degree:** Master’s Thesis, Politecnico di Torino  
**Affiliation:** Università degli Studi dell’Insubria  

---

## Overview

This project implements an **AI agent for dermatological diagnosis** that combines **Large Language Models (LLMs)** and **vision-based tools** to perform multimodal reasoning, skin lesion analysis, and explainable decision-making.

The framework integrates:
- **Classification Tool** – a Swin Transformer–based vision model for multiclass dermatological image classification.  
- **Explanation Tool** – generates interpretable saliency maps using techniques like Vanilla Gradient, Integrated Gradients, and Grad-CAM.  
- **Detection Tool** – a Mask DINO-based segmentation module for panoramic lesion localization.  
- **Central Agent** – an LLM-orchestrated reasoning engine that dynamically calls these tools using a ReAct loop, imitating clinical workflows.

The agent enables **transparent, interpretable, and modular diagnostic reasoning**, supporting clinicians in real-world dermatology settings.

---
## Environment Setup

All dependencies are listed in the `environment.yml` file.  
To create the environment:

```bash
conda env create -f environment.yml
conda activate dermatology-agent
```
Once activated, configure the environment variables inside the file named model_env.env in the project root directory.

⚠️ The system requires a valid OpenAI API key to enable multimodal reasoning via the LLM interface.

## Main Interface

To launch the graphical user interface (GUI) for the dermatology agent:

```bash
python main.py
```
The agent include three main tools:
- **Classification Tool**: Classifies skin lesion images using a Swin Transformer
- **Explanation Tool**: Generates saliency maps to explain classification decisions using Vanilla Gradient
- **Detection Tool**: Detects skin lesions in images using Mask DINO

⚠️ The classification tool requires a pre-trained model checkpoint. You can make a request to the author to obtain it.
Send the request to: lsgroi@studenti.uninsubria.it.

⚠️ The detection tool requires a pre-trained Mask DINO model checkpoint. You can download it from the official repository:
follow the instructions here:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install .
cd ..

git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..
```

Then, download the weights from Kaggle:

```bash
kaggle datasets download -d resimasss/maskdino-weights --unzip
```
### Notes

- You may need to update dataset registration paths in the scripts.
- For custom datasets, refer to [Detectron2's dataset registration guide](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html).
- Verify that your GPU drivers, PyTorch, and CUDA versions are compatible.

---

### License

This project is licensed under the **Apache 2.0 License**, inherited from:

- [Mask DINO](https://github.com/IDEA-Research/MaskDINO)
- [Detectron2](https://github.com/facebookresearch/detectron2)
