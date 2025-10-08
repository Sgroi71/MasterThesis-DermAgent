import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from DermAgent.agent import *
from DermAgent.tools import *
from DermAgent.utils import *


logging.set_verbosity_error()

# --------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------
if not load_dotenv():
    print("Warning: .env file not found, using default values")

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "1")
DEVICE = os.getenv("DEVICE", "mps")
ROOTP = os.getenv("ROOTP")
ROOT = os.getenv("ROOT")

# Server configuration
SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8585"))

# LLM configuration
MODEL = os.getenv("MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
PROMPT_FILE = os.getenv("PROMPT_FILE", "DermAgent/docs/system_prompts.txt")

# Detection tool configuration
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
CONFIG_PATH_DET = os.getenv("CONFIG_PATH_DET")
SCORE_TH = float(os.getenv("SCORE_TH", "0.4"))
BOX_FORMAT = os.getenv("BOX_FORMAT", "xyxy")

# Classification tool configuration
CONFIG_PATH = os.getenv("CONFIG_PATH")
OUTPUT_HEAD = os.getenv("OUTPUT_HEAD", "All")



def initialize_agent(
    prompt_file,
    tools_to_use=None,
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the DermAgent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "MuteClassifierTool": lambda: MuteClassifierTool(
                                        device = device,
                                        config_path = CONFIG_PATH,
                                        output_head = OUTPUT_HEAD
                                    ),
        "ExplainationTool": lambda: ExplanationTool(
                                        device = device,
                                        config_path = CONFIG_PATH
                                    ),  
        "DINODetectionTool": lambda: DINODetectionTool(
                                        device=device,
                                        config_path=CONFIG_PATH_DET,
                                        weights_path=WEIGHTS_PATH,
                                        score_threshold=SCORE_TH
                                    )
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the DermAgent application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    selected_tools = [
        "MuteClassifierTool",
        "ExplainationTool",
        "DINODetectionTool"
    ]
    if not load_dotenv(f"{ROOT}/model_env.env"):
        print(f"Error loading environment variables from model_env.env")
        exit(1)

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key


    agent, tools_dict = initialize_agent(
        prompt_file = PROMPT_FILE,
        tools_to_use = selected_tools,
        device = DEVICE,   # or "cpu"
        model = MODEL,  
        temperature = TEMPERATURE,
        top_p = TOP_P,
        openai_kwargs = openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=True)
