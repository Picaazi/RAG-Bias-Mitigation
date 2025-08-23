import wandb
from pipeline import pipeline, data_router
import numpy as np
import random
from dotenv import load_dotenv
import os
import argparse
import torch

# # Load API keys 
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")

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
    {
        "dataset": "gender_bias",
        "mode": "both"
    },
]

# Set random seed 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
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
        print(f"\nRunning experiment for:")
        print(f"dataset={cfg['dataset']}")
        print(f"mode={cfg['mode']}")
        print(f"seed={cfg['seed']}")
        print(f"top_k={cfg['top_k']}")
        print(f"result_folder={cfg['result_folder']}")
        
        set_seed(cfg["seed"])  # Set seed for reproducibility
        #Comment out or delete to determine consistent results or randomness
        #set_seed(cfg["seed"])

        # Load dataset
        questions, docs = data_router(cfg["dataset"])
        questions = questions[:2]  # add a limit for number of questions of the dataset

        # Initialize WandB
        wandb.init(project="bias-mitigation", config=cfg)

        # Run pipeline (pipeline already saves CSV)
        pipeline(
            questions=questions,
            docs=docs,
            k=cfg["top_k"],
            mode=cfg["mode"],
            result_folder=cfg["result_folder"]
        )

        # Find the most recent CSV file saved by pipeline
        saved_files = sorted(
            [f for f in os.listdir(cfg["result_folder"]) if f.startswith(f"results_{cfg['mode']}")],
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
