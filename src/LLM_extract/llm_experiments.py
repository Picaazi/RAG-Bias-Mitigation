import wandb
from pipeline import pipeline, data_router
import numpy as np
import random
from dotenv import load_dotenv
import os
import openai
from LLMversion_pipeline import llm_pipeline

#For me to use
env_path = os.path.join(os.path.dirname(__file__), "api.env")
load_dotenv(env_path)

openai.api_key = os.environ.get("OPENAI_KEY")

if openai.api_key is None:
    raise ValueError("OPENAI_KEY not found. Make sure api.env is in the src folder.")
else:
    print("OPENAI_KEY loaded successfully")

# # Load API keys 
# load_dotenv()  
# api_key = os.getenv("OPENAI_KEY")

# Create results folder for csv files for local use 
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Fixed configuration
fixed_var_config = {
    "seed": None, #change for randomness or consistent results, (None = truly random, number = deterministic).
    "top_k": 5,
    "result_folder": RESULTS_FOLDER  
}

# Experiment configurations - configs can be added
experiment_configs = [
    {"dataset": "gender_bias", "mode": "rewrite"},
]

# Set random seed 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# Main experiment loop
def run_all_experiments():
    for exp_cfg in experiment_configs:
        cfg = {**fixed_var_config, **exp_cfg}
        print(f"\n=== Running dataset={cfg['dataset']} mode={cfg['mode']} ===")

        # Load dataset
        questions, docs = data_router(cfg["dataset"])
        questions = questions[:2]  # limit for testing

        # Initialize WandB
        wandb.init(project="bias-mitigation", config=cfg)

        # Run pipeline (pipeline already saves CSV)
        llm_pipeline(
            questions=questions,
            docs=docs,
            k=cfg["top_k"],
            mode=cfg["mode"]
        )

        # Find the most recent CSV file saved by pipeline
        saved_files = sorted(
            [f for f in os.listdir(RESULTS_FOLDER) if f.startswith(f"results_with_bias_{cfg['mode']}")],
            key=lambda x: os.path.getmtime(os.path.join(RESULTS_FOLDER, x))
        )

        if saved_files:
            csv_path = os.path.join(RESULTS_FOLDER, saved_files[-1])
            print(f"Logging CSV to WandB: {csv_path}")
            artifact = wandb.Artifact(name=f"results_{cfg['dataset']}_{cfg['mode']}", type="dataset")
            artifact.add_file(csv_path)
            wandb.log_artifact(artifact)

        wandb.finish()

if __name__ == "__main__":
    run_all_experiments()
