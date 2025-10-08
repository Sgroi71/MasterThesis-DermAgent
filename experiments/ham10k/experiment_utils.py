import json
import re
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List
import csv

def retrive_probabilities_from_logs(logs_path, image_name):
    """
    Retrieve the probabilities of the classes from the logs of the Mute Classifier.

    Args:
        logs_path: Path to the directory containing the logs.
        image_name: Name of the image.
        predicted_class: Predicted class.

    Returns:
        A dictionary with the class names and their corresponding probabilities.
    """
    logs = glob.glob(os.path.join(logs_path, "*.json"))
    for l in logs:
        with open(l, 'r') as file:
            mute_answer = json.load(file)[0]
            if mute_answer['args']['image_path'].split("/")[-1][:-4] == image_name:
                return estract_mute_answer(mute_answer['content'])[1]
    return None

def estract_mute_answer(mute_answer):
    """
    Extract the class with the highest probability and map it to the corresponding label, 
    along with the probability of that class.

    Args:
        mute_answer: A dictionary containing probabilities for each class and additional metadata.

    Returns:
        A tuple containing the mapped class label with the highest probability and its probability.
    """
    class_mapping = {
        "Melanoma": "MEL",
        "Melanocytic Nevus": "NV",
        "Basal Cell Carcinoma": "BCC",
        "Actinic Keratosis / Intraepithelial Carcinoma": "AKIEC",
        "Benign Keratosis-like Lesions": "BKL",
        "Dermatofibroma": "DF",
        "Vascular Lesions": "VASC",
    }
    # Transform from string to tuple of two dictionaries
    mute_answer = eval(mute_answer)
    probabilities, _ = mute_answer
    if "HAM10k" in probabilities:
        probabilities = probabilities["HAM10k"]
    if not isinstance(probabilities, dict):
        raise ValueError("Invalid mute_answer format. Expected a dictionary for probabilities.")

    # Find the class with the highest probability
    max_class = max(probabilities, key=probabilities.get)
    max_probability = probabilities[max_class]
    return class_mapping.get(max_class, "Unknown"), max_probability

def evaluate_gpt4o_reliability(file_path_results,log_path,benchmark_gt_file):
    """
    Evaluate the reliability of the model by checking if the answers of gpt4o are consistent with Mute Classifier.

    Args:
        file_path_results: Path to the JSON file with the results.
        log_path: Path to the directory where the logs will be saved.
    """

    GT_dict,_= read_csv_to_dict_Metadata(benchmark_gt_file)
    # Load the results from the JSON file
    with open(file_path_results, 'r') as file:
        gpt_answers = json.load(file)
    gpt_answers = {k: v for d in gpt_answers for k, v in d.items()}
    logs = glob.glob(os.path.join(log_path, "*.json"))
    corrects=0
    wrongs=0
    bads=0
    for l in logs:
        with open(l, 'r') as file:
            mute_answer = json.load(file)[0]
        
        image_name=mute_answer['args']['image_path'].split("/")[-1]
        # Check if the image name exists in the gpt_answer
        if image_name in gpt_answers:
            if estract_mute_answer(mute_answer['content'])[0]==gpt_answers[image_name]:
                corrects+=1
            else:
                wrongs+=1
                print(f"Image {image_name} has different answers: final->{gpt_answers[image_name]} vs Mute->{estract_mute_answer(mute_answer['content'])[0]} vs Correct->{GT_dict[image_name[:-4]]}")
        else :
            print(f"Image {image_name} not found in gpt answers.")
            bads+=1
            continue
    total = corrects + wrongs + bads
    accuracy = corrects / total if total > 0 else 0
    print("Evaluation of the reliability of the agent:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Corrects: {corrects}")
    print(f"Wrongs: {wrongs}")
    print(f"Bads: {bads}")

def count_objects(file_path):
    """
    Reads a JSON or CSV file and counts the number of objects or rows in it.

    Args:
        file_path: Path to the JSON or CSV file.

    Returns:
        int: Number of objects in the JSON file or rows in the CSV file.
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                count = len(json_data)
                return count
        elif file_path.endswith('.csv'):
            with open(file_path, 'r') as file:
                count = sum(1 for line in file) - 1  # Subtract 1 for the header
                return max(count, 0)  # Ensure non-negative count
        else:
            print("Unsupported file format. Please provide a JSON or CSV file.")
            return 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0

def process_data(data):
    """
    Process the answer and return the indended one ('MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC') or -1 in case of error.
    
    Args:
    data: Answer to be processed

    Returns:
    answer: Processed answer
    """
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    for cls in classes:
        if re.search(rf'\b{cls}\b', data):
            return cls
    with open('wrong_answers.txt', 'a') as file:
        file.write(data + '\n')
    return -1

def process_data_binary(data):
    """
    Process the answer and return the indended one ('1', '0') or -1 in case of error.

    Args:
    data: Answer to be processed
    Returns:
    answer: Processed answer
    """
    classes = [1, 0]
    searched=['YES','NO']
    for i, cls in enumerate(classes):
        if re.search(rf'\b{searched[i]}\b', data.upper()):
            return cls
    with open('wrong_answers.txt', 'a') as file:
        file.write(data + '\n')
   
    return -1

def read_csv_to_dict_Metadata(csv_file_path,year=2018,test=False):
    """
    Reads a CSV file and converts it into two dictionaries:
    1. A dictionary with image_id as keys and dx as values.
    2. A dictionary with image_id as keys and a nested dictionary containing age, sex, and localization as values.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]: 
        - A dictionary mapping image_id to dx if year=2018.
        - A dictionary mapping image_id to a nested dictionary with age, sex, and localization.
    """
    image_dx_dict = {}
    image_metadata_dict = {}
    number_of_lines = 8 if year == 2018 else 5 if not test else 4 # Number of expected columns in the CSV file

    try:
        with open(csv_file_path, 'r') as csvfile:
            for idx, line in enumerate(csvfile):
                if idx == 0:  # Skip the header line
                    continue
                line = line.strip()
                parts = line.split(',')
                if len(parts) < number_of_lines:
                    raise ValueError(f"Malformed line in CSV: {line}")
                
                if year == 2018:
                    # For HAM10000 dataset
                    _, image_id, dx, _, age, sex, localization, _ = parts
                    image_dx_dict[image_id] = dx.upper()
                    image_metadata_dict[image_id] = {
                        "age": float(age) if age else None,
                        "sex": sex if sex!="unknown" else None,
                        "localization": localization if localization!="unknown" else None
                    }
                elif year == 2019:
                    if not test:
                        image_id, age, localization, _, sex = parts
                    else:
                        image_id, age, localization, sex = parts
                    image_metadata_dict[image_id] = {
                        "age": float(age) if age!="" else None,
                        "sex": sex if sex!="" else None,
                        "localization": localization if localization!="" else None
                    }
                    

            
        return image_dx_dict, image_metadata_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")

def create_gt_dict(dataset_path):
    """
    Reads creates a dictionary mapping image names to their respective classes.

    Args:
    dataset_path: Path to the dataset directory
    Returns:
    A dictionary with image names as keys and class names as values
    """
    image_class_dict = {}
    
    Ham10k = glob.glob(os.path.join(dataset_path, "*.jpg"))
    Image_net = glob.glob(os.path.join(dataset_path, "*.JPEG"))
    
    for image_path in Ham10k:
        image_name = os.path.basename(image_path)[:-4]
        image_class_dict[image_name] = 1  # 1 for Ham10k

    for image_path in Image_net:
        image_name = os.path.basename(image_path)[:-5]  # Remove the .JPEG extension
        image_class_dict[image_name] = 0  # 0 for ImageNet

    return image_class_dict




def postProcessingResults(file_path_results, benchmark_gt_file_path, benchmark_binary_directory, classification_type="multi-class", use_tools=False):
    """
    print the accuracy of the results compared to the ground truth.
    
    Args:
    file_path: Path to the JSON file

    Returns:
    accuracy: Accuracy of the predictions
    """
    if classification_type=="multi-class":
        GT_dict,_= read_csv_to_dict_Metadata(benchmark_gt_file_path)
        corrects=0
        wrongs=0
        bads=0
        try:
            with open(file_path_results, 'r') as file:
                data = json.load(file)
                for item in data:
                    for image_name, predicted_class in item.items():
                        if predicted_class == "Error" :
                            bads+=1
                        else:
                            predicted_class = process_data(predicted_class)
                            if predicted_class == -1:
                                bads+=1
                                continue
                            if image_name.endswith(".jpg"):
                                image_name = image_name[:-4]  # Remove the .jpg extension
                            if image_name in GT_dict:
                                if GT_dict[image_name] == predicted_class:
                                    corrects += 1
                                else:
                                    wrongs += 1
                            else:
                                print(f"Image {image_name} not found in ground truth.")
            total = corrects + wrongs + bads
            accuracy = corrects / total if total > 0 else 0
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Corrects: {corrects}")
            print(f"Wrongs: {wrongs}")
            print(f"Bads: {bads}")
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return -1
    elif classification_type=="binary":
        GT_dict = create_gt_dict(benchmark_binary_directory)
        if use_tools:
            corrects_tool = 0
            wrongs_toll = 0
            corrects_YES =0
            wrongs_YES = 0
            bads = 0

            y_pred_tool = []
            y_pred_YES = []
            y_true= []

            try:
                with open(file_path_results, 'r') as file:
                    data = json.load(file)
                    for item in data:
                        image_name= item["image_name"]
                        predicted_class= item["predicted_class"]
                        used=item["use_tools"]

                        predicted_class_tool = 1 if used else 0
                        if predicted_class == "Error":
                            bads += 1
                            predicted_YES = -1
                        else:
                            predicted_YES= process_data_binary(predicted_class) if process_data_binary(predicted_class)>=0 else 0 #if the answer is -1, means that the agent answer doesn't contain YES or NO, so since we didn't specify when saying No, we assume that the answer is NO

                        if image_name.endswith(".jpg"):
                            image_name = image_name[:-4]  # Remove the .jpg extension
                        elif image_name.endswith(".JPEG"):
                            image_name = image_name[:-5]

                        if image_name in GT_dict:
                            y_true.append(int(GT_dict[image_name]))
                            y_pred_tool.append(int(predicted_class_tool))

                            if predicted_YES > -1:
                                y_pred_YES.append(int(predicted_YES))
                                if int(GT_dict[image_name]) == int(predicted_YES):
                                    corrects_YES += 1
                                else:
                                    wrongs_YES += 1
                            
                            if int(GT_dict[image_name]) == int(predicted_class_tool):
                                corrects_tool += 1
                            else:
                                wrongs_toll += 1
                        else:
                            print(f"Image {image_name} not found in ground truth.")
                    
                total_tool = corrects_tool + wrongs_toll
                accuracy_tool = corrects_tool / total_tool if total_tool > 0 else 0
                print ("Binary classification results (based on how many times the tool was used):")
                print(f"Accuracy: {accuracy_tool:.4f}")
                print(f"Corrects: {corrects_tool}")
                print(f"Wrongs: {wrongs_toll}")

                total_YES = corrects_YES + wrongs_YES + bads
                accuracy_YES = corrects_YES / total_YES if total_YES > 0 else 0
                print ("Binary classification results (based on how many times the model answer YES):")
                print(f"Accuracy: {accuracy_YES:.4f}")
                print(f"Corrects: {corrects_YES}")
                print(f"Wrongs: {wrongs_YES}")
                print(f"Bads: {bads}")


                # Generate confusion matrix
                if y_true and y_pred_tool:
                    cm = confusion_matrix(y_true, y_pred_tool, labels=[1, 0])
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham10k (1)", "ImageNet (0)"])
                    disp.plot(cmap=plt.cm.Blues)
                    plt.title("Confusion Matrix (based on how many times the tool was used)")
                    plt.show()
                if y_true and y_pred_YES:
                    cm = confusion_matrix(y_true, y_pred_YES, labels=[1, 0])
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham10k (1)", "ImageNet (0)"])
                    disp.plot(cmap=plt.cm.Blues)
                    plt.title("Confusion Matrix (based on how many times the model answer YES)")
                    plt.show()
            except Exception as e:
                print(f"Error reading JSON file: {e}")
                return -1

                        

        else:
            corrects = 0
            wrongs = 0
            y_true = []
            y_pred = []
            bads = 0

            try:
                with open(file_path_results, 'r') as file:
                    data = json.load(file)
                    for item in data:
                        
                        image_name, predicted_class = list(item.items())[0]
                        
                        if predicted_class == "Error":
                            bads += 1
                        else:
                            predicted_class = process_data_binary(predicted_class)
                            if predicted_class == -1:
                                bads += 1
                                continue
                            if image_name.endswith(".jpg"):
                                image_name = image_name[:-4]  # Remove the .jpg extension
                            elif image_name.endswith(".JPEG"):
                                image_name = image_name[:-5]
                            if image_name in GT_dict:
                                y_true.append(int(GT_dict[image_name]))
                                y_pred.append(int(predicted_class))
                                
                                if int(GT_dict[image_name]) == int(predicted_class):
                                    corrects += 1
                                else:
                                    wrongs += 1
                            else:
                                print(f"Image {image_name} not found in ground truth.")
                total = corrects + wrongs + bads
                accuracy = corrects / total if total > 0 else 0
                print ("Binary classification results (based on how many times the model answer YES):")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Corrects: {corrects}")
                print(f"Wrongs: {wrongs}")
                print(f"Bads: {bads}")
                # Generate confusion matrix
                if y_true and y_pred:
                    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham10k (1)", "ImageNet (0)"])
                    disp.plot(cmap=plt.cm.Blues)
                    plt.title("Confusion Matrix")
                    plt.show()
            except Exception as e:
                print(f"Error reading JSON file: {e}")
                return -1
    else:
        print("Error: classification type not supported")
        return -1

def list_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out = []
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                out.append(os.path.join(dp, f))
    out.sort()
    return out


def ts_rome_iso() -> str:
    return datetime.now(ZoneInfo("Europe/Rome")).isoformat(timespec="microseconds")


def load_gt_onehot_csv(csv_path: str) -> Dict[str, str]:
    """
    CSV format you provided:
    image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
    ISIC_0034524,0.0,1.0,0.0,0.0,0.0,0.0,0.0

    Returns: dict base_name_without_ext -> HAM10k textual label
    """
    # HAM10k label order in your header
    ham_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    

    gt: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # sanity check
        for col in ["image", *ham_cols]:
            if col not in r.fieldnames:
                raise ValueError(f"Missing column '{col}' in {csv_path}")

        for row in r:
            base = row["image"]
            # values as floats
            vals = [float(row[c]) for c in ham_cols]
            if not any(v != 0.0 for v in vals):
                # all zeros -> unknown
                gt[base] = None
                continue
            # argmax
            k = max(range(len(vals)), key=lambda i: vals[i])
            gt[base] = ham_cols[k]
    return gt

def load_gt_dx_csv(csv_path: str) -> Dict[str, str]:
    """
    CSV format:
    lesion_id,image_id,dx,dx_type,age,sex,localization,dataset
    HAM_0000118,ISIC_0027419,bkl,histo,80.0,male,scalp,vidir_modern

    Returns: dict base_name_without_ext -> dx label (e.g. 'bkl')
    """
    gt: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "image_id" not in r.fieldnames or "dx" not in r.fieldnames:
            raise ValueError(f"Missing required columns in {csv_path}")
        for row in r:
            base = row["image_id"]
            gt[base] = row["dx"].upper()
    return gt

def topk_from_head(head_dict: Dict[str, float], k: int = 3):
    return sorted(head_dict.items(), key=lambda kv: float(kv[1]), reverse=True)[:k]


def basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
    
def plot_error_probabilities(error_logs_path,name):
    """
    Plot an histogram of the probabilities of the classes for the wrong predictions.
    Args:
    error_logs_path: Path to the JSON file with the error logs.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json

    with open(error_logs_path, 'r') as f:
        error_logs = json.load(f)

    probabilities = []
    for log in error_logs:
        if log["class_probability"]:
            probabilities.append(log["class_probability"])
    probabilities = [prob for prob in probabilities if isinstance(prob, (int, float))]  # Filter out non-numeric values
    probabilities = np.array(probabilities)
    # Plot the histogram
    plt.hist(probabilities, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Class Probabilities for Wrong Predictions')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'error_probabilities_{name}.png')