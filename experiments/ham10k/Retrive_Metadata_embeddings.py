import openai
import os
import glob
import json
from dotenv import load_dotenv
from openai import OpenAI
import sys
from google import genai
from google.genai import types
import time
ROOT = ... # path to the root of the project
sys.path.append(ROOT)

from experiments.Ham10k.experiment_utils import read_csv_to_dict_Metadata
from tqdm import tqdm


METADATA_FILE = f"{ROOT}/datasets/ISIC2019_Metadata/ISIC_2019_Test_Metadata.csv"



def get_embeddings(prompt, openai_kwargs):
    """
    Retrieve embeddings for the given prompt using OpenAI API.

    Args:
        prompt (str): The input text for which embeddings are to be retrieved.
        api_keys (dict): A dictionary containing the OpenAI API key with the key 'openai_api_key'.

    Returns:
        list: A list of embeddings for the input prompt.
    """
    if openai_kwargs["model"]== "text-embedding-3-small":
        client = OpenAI(api_key=openai_kwargs["api_key"])
        try:
            response=client.embeddings.create(
                model=openai_kwargs["model"],
                input=prompt,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"An error occurred while retrieving embeddings: {e}")
            return None
    elif openai_kwargs["model"]== "text-embedding-004":
        client = genai.Client(api_key=openai_kwargs["api_key"])

        result = client.models.embed_content(
                model=openai_kwargs["model"],
                contents=prompt,
                config=types.EmbedContentConfig(task_type="CLASSIFICATION")
        )
        return result.embeddings[0].values
def create_input(metadata):
    """
    Create a discursive input string for the OpenAI API based on the metadata.

    Args:
        metadata (dict): The metadata dictionary containing information about the image.

    Returns:
        str: A natural-language description of the patient and lesion.
    """
    parts = []

    if metadata.get("age"):
        parts.append(f"The patient is {int(metadata['age'])} years old")

    if metadata.get("sex"):
        
        parts.append(f"{'who is ' if len(parts) == 1 else 'Patient is '}{metadata['sex'].lower()}")

    if metadata.get("localization"):
        parts.append(f"{'with' if parts else 'There is'} a skin lesion located on the {metadata['localization']}")

    return " ".join(parts) + "." if parts else ""




def retrive_embeddings(output_file, openai_kwargs, n_samples=10, year=2018, test=False):
    """
    Retrieve embeddings for metadata entries and save them to a JSON file.
    Args:
        output_file (str): The path to the output JSON file.
        openai_kwargs (dict): A dictionary containing OpenAI API parameters.
        n_samples (int): The number of samples to process. Default is 10.
        year (int): The year of the metadata to process. Default is 2018.
        test (bool): Whether to use the test set or not. Default is False.
    """
    _, metadata = read_csv_to_dict_Metadata(METADATA_FILE, year=year, test=test)
    count = 0
    if n_samples > len(metadata):
        n_samples = len(metadata)
    metadata = dict(list(metadata.items())[:n_samples])
    already_processed = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
            results = existing_data
            already_processed = {item["image_name"] for item in existing_data}
            print(f"Already processed {len(existing_data)} images")
            n_samples = len(existing_data) + n_samples
    i=0
    for image_name, metadata in tqdm(metadata.items(), desc="Processing images"):
        if metadata["age"] or metadata["sex"] or metadata["localization"]:
            if image_name in already_processed:
                continue
            input = create_input(metadata)
            embeddings = get_embeddings(input, openai_kwargs)
            time.sleep(0.43)  # Limita a 140 richieste/minuto
            if embeddings:
                result = {
                    "model": openai_kwargs["model"],
                    "image_name": image_name,
                    "input": input,
                    "embeddings": embeddings
                }
                results.append(result)
            else:
                print(f"Failed to retrieve embeddings for {image_name}")
                result = {
                    "model": openai_kwargs["model"],
                    "image_name": image_name,
                    "input": input,
                    "error": "Failed to retrieve embeddings"
                }
                results.append(result)
        else:
            count += 1
            result = {
                "model": openai_kwargs["model"],
                "image_name": image_name,
                "input": None,
                "error": "No metadata available"
            }
            results.append(result)
        if i%500==0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
        i += 1
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Skipped {count} images with no metadata.")

def count_metadata(file_path):
    """
    Reads a JSON or CSV file and counts the number of objects or rows in it.

    args:
        file_path (str): Path to the JSON or CSV file.
    returns:
        int: Number of objects in the file.
    """
    dimMetadata=[]
    count=0
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                try:
                    l=json.load(file)
                    print (f"Number of objects in JSON file: {len(l)}")
                    for d in l:
                        if "embeddings" in d:
                            dimMetadata.append(len(d["embeddings"]))
                            if len(d["embeddings"])==3072:
                                print(f"Found valid embedding for {d['image_name']} with dimension {len(d['embeddings'])}")
                                count+=1
                        else:
                            print(f"Skipping entry without 'embeddings': {d}")
                            # If you want to count the number of embeddings, uncomment the next line
                            count += 1
                        #dimMetadata.append(len(d["embeddings"]))
                        #count += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
                print (f"same dim for all the embeddings: {len(set(dimMetadata))==1} with dim {dimMetadata[0]}: {set(dimMetadata)}")
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

    
if __name__ == "__main__":
    model= "Gemini"  # gpt or Gemini
    if not load_dotenv(f"{ROOT}/{model}_env.env"):
        print(f"Error loading environment variables from {model}_env.env")
        exit(1)
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    #openai_kwargs["model"] = "text-embedding-3-small"
    openai_kwargs["model"] = "text-embedding-004"
    
    output_file = os.path.join(ROOT, "experiments","results","Metadata_2019", "Metadata_embeddings_test_2019_Gemini.json")

    print (count_metadata(output_file))
    retrive_embeddings(output_file, openai_kwargs, n_samples=2000000, year=2019, test=True)