# run_dataset_mute_classifier.py
from tqdm import tqdm
import json
import os
import torch
from PIL import Image
import sys

ROOT = ... # path to the root of the project
sys.path.append(ROOT)

from DermAgent.agent import *            
from DermAgent.tools import *            
from DermAgent.utils import *  
from experiments.Ham10k.experiment_utils import *          

BENCHMARK_DIR = ... # path to the benchmark image directory HAM10k
BENCHMARK_GT_FILE = f"{ROOT}/datasets/ISIC2018_Task3_Metadata/HAM10000_metadata.csv"


# ---------- main ----------
def main():
    conf_path=ROOT+"/checkpoints/exp-HAM+Derm7pt-all+BCN+HAM-bin+DermNet+Fitzpatrick.yaml"
    device="cuda"
    head="All"
    output_log=f"{ROOT}/logs/"
    output_summary_json=f"{ROOT}/experiments/results/Ham10kClassification/HAM10k_Mute_summary_2.json"

    ham_map = {
        "Melanoma": "MEL",
        "Melanocytic Nevus": "NV",
        "Basal Cell Carcinoma": "BCC",
        "Actinic Keratosis / Intraepithelial Carcinoma": "AKIEC",
        "Benign Keratosis-like Lesions": "BKL",
        "Dermatofibroma": "DF",
        "Vascular Lesions": "VASC",
    }
    

    # init tool (from medDerm.tools import *)
    tool = MuteClassifierTool(
        pretrained=False,
        device=device,
        config_path=conf_path,
        output_head="All",
    )

    # enumerate images
    images = list_images(BENCHMARK_DIR)
    if not images:
        print(f"[WARN] No images found in {BENCHMARK_DIR}")
        return

    # load GT (one-hot to textual HAM10k label)
    gt_map = load_gt_dx_csv(BENCHMARK_GT_FILE)

    tool_log = []
    summary = []

    for img in tqdm(images):
        try:
           
            x = Image.open(img).convert("RGB")
            with torch.inference_mode():
                tensor = tool.transform(x).unsqueeze(0).to(tool.device)
                outputs = tool.model.forward_classification_tasks(tensor)

            # build the multi-head dict like your _run
            total_outputs = {}
            if tool.head != "All":
                head = tool.head
                preds = outputs[head]["predictions"].tolist()
                labels = tool.model.dataset_info[head]["class_labels"]
                total_outputs[head] = dict(zip(labels, preds))
            else:
                for h, hdict in outputs.items():
                    preds = hdict["predictions"].tolist()
                    labels = tool.model.dataset_info[h]["class_labels"]
                    total_outputs[h] = dict(zip(labels, preds))

            metadata = {
                "image_path": img,
                "analysis_status": "completed",
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }

            # --- tool-style log entry (content is a tuple string) ---
            tool_log.append(
                {
                    "name": "Mute_classifier",
                    "args": {"image_path": img},
                    "content": repr((total_outputs, metadata)),
                    "timestamp": ts_rome_iso(),
                }
            )
            namefile = os.path.basename(img).split(".")[0]
            output_log_path = os.path.join(output_log, f"{namefile}_Mute_log.json")
            with open(output_log_path, "w", encoding="utf-8") as f:
                json.dump(tool_log, f, ensure_ascii=False, indent=2)

            # --- clean summary row ---
            base_no_ext = basename_no_ext(img)
            gt_label = gt_map.get(base_no_ext)

            if isinstance(total_outputs, dict) and "HAM10k" in total_outputs and isinstance(total_outputs["HAM10k"], dict):
                ham_top3 = [{"class": ham_map[c], "prob": float(p)} for c, p in topk_from_head(total_outputs["HAM10k"], k=3)]
            else:
                ham_top3 = None

            summary.append(
                {
                    "image_name": os.path.basename(img),
                    "image_path": img,
                    "ground_truth": gt_label,
                    "predicted_class": ham_top3[0]["class"] if ham_top3 else None,
                    "top3_ham": ham_top3,       
                    "metadata": metadata,
                }
            )

        except Exception as e:
            err_meta = {"image_path": img, "analysis_status": "failed", "error": str(e)}
            tool_log.append(
                {"name": "Mute_classifier", "args": {"image_path": img}, "content": repr(({"error": str(e)}, err_meta)), "timestamp": ts_rome_iso()}
            )
            summary.append(
                {
                    "image_name": os.path.basename(img),
                    "image_path": img,
                    "ground_truth": gt_map.get(basename_no_ext(img)),
                    "top3_ham": None,
                    "error": str(e),
                    "metadata": err_meta,
                }
            )

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {len(tool_log)} entries -> {output_log}")
    print(f"[DONE] {len(summary)} entries -> {output_summary_json}")


if __name__ == "__main__":
    main()