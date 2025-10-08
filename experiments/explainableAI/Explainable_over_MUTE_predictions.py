import json
import os
import torch
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
ROOT=...
sys.path.append(ROOT)
from DermAgent.tools.explanation import ExplanationTool
if __name__ == "__main__":
    class_mapping = {
        "MEL":"Melanoma",
        "NV":"Melanocytic Nevus" ,
        "BCC":"Basal Cell Carcinoma",
        "AKIEC":"Actinic Keratosis / Intraepithelial Carcinoma",
        "BKL":"Benign Keratosis-like Lesions",
        "DF":"Dermatofibroma",
        "VASC":"Vascular Lesions",
    }

    # Load the explanation tool
    explanation_tool = ExplanationTool(
        device="cpu",
        config_path=f"{ROOT}/checkpoints/exp-HAM+Derm7pt-all+BCN+HAM-bin+DermNet+Fitzpatrick.yaml",   
    )
    head="HAM10k"

    # Define the input for the explanation tool
    with open(f"{ROOT}/experiments/results/Ham10kClassification/HAM10k_Mute_summary_3.json", "r") as f:
        images = json.load(f)
    
    print(f"samples to execute: {len(images)}")
    IMAGE_DIR = f"{ROOT}/datasets/ISIC2018_Task3_Test_Input/"
    OUTPUT_DIR = f"{ROOT}/experiments/results/Explainable/thesis/"
    explainable_techniques = [
        "integrated_gradients",
        "occlusion",
        "input_x_gradient",
        "vanilla_gradient",
        "shap"
    ] 
    
    total_res={k:[] for k in explainable_techniques}

    # used for the smoothgrad experiments
    # total_res["integrated_gradients_SG"]=[]
    # total_res["vanilla_gradient_SG"]=[]
    # total_res["input_x_gradient_SG"]=[]
    for l in tqdm(images, desc="Processing images"):
        if l["image_name"]+".jpg" not in os.listdir(IMAGE_DIR) and l["image_name"] not in os.listdir(IMAGE_DIR):
            print(f"Image {l['image_name']} not found in the dataset.")
            continue
        for explanation_technique in explainable_techniques:
            print(f"Running explanation for {l['image_name']}.jpg using {explanation_technique} technique.")
            
            input_data = {
                "input_img_path": l["image_path"],
                "explanation_technique": explanation_technique,
                "targetClass":class_mapping[ l["predicted_class"]],
                "head": "HAM10k",
                "SG": False,
                "alpha_overlay": 0.5,
                "output_image_path": OUTPUT_DIR
            }
            # Run the explanation tool
            output,_=explanation_tool._run(**input_data)
            # A different version of the explanation tool returns also the IoU value between saliency map and GT segmentation
            #total_res[explanation_technique].append(output["IoU"])

            if explanation_technique in ["integrated_gradients", "input_x_gradient", "vanilla_gradient"]: #these strategy are analized with and without SG
                # Run the explanation tool
                input_data["SG"]=True
                output,_=explanation_tool._run(**input_data)
                total_res[f"{explanation_technique}_SG"].append(output["IoU"])
            

            
            # used to evaluate the IoU between saliency map and GT segmentation using target label=ground truth
            if l["predicted_class"]!= l["ground_truth"]:
                input_data["targetClass"]= class_mapping[l["ground_truth"]]
                input_data["output_image_path"]=f"{ROOT}/experiments/results/Explainable/160WrongShapExplanations/"
                #IoUs_Wrong_pred.append(output["IoU"])
                output,_=explanation_tool._run(**input_data)
                #IoUs_Wrong_gt.append(output["IoU"])
            else:
                #IoUs_Correct.append(output["IoU"])
                ...
    filepath = os.path.join(ROOT, "experiments","results", "Explainable", "summaryResults_techniques.json")
    with open(filepath, "w") as f:
        json.dump(total_res, f, indent=4) 
    print(f"mIoU over 320 predictions (160 correct, 160 wrongs) between saliecy map and GT segmentation using target label=predicted class:")
    for explanation_technique in total_res:
        print(f"{explanation_technique}---->{sum(total_res[explanation_technique])/len(total_res[explanation_technique])}")
    

           
       
        

            
            
            