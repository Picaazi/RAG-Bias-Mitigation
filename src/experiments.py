import wandb
from pipeline import pipeline, data_router
import numpy as np
import random
from dotenv import load_dotenv
import os
import argparse
import torch

# Load API keys 
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")

# Create results folder for csv files for local use 
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Set random seed 
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

# Main experiment function
def run_experiment(args):
    cfg = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "top_k": args.top_k,
        "result_folder": RESULTS_FOLDER
    }
    
    print(f"\nRunning experiment with:")
    print(f"dataset={cfg['dataset']}")
    print(f"mode={cfg['mode']}")
    print(f"seed={cfg['seed']}")
    print(f"top_k={cfg['top_k']}")
    print(f"num_questions={args.num_questions}")
    print(f"result_folder={cfg['result_folder']}")
    
    set_seed(cfg["seed"])

    # Load dataset
    questions, docs = data_router(cfg["dataset"])
    questions = questions[:args.num_questions]

    # Initialize WandB
    wandb.init(project=args.project_name, config=cfg)

    # Run pipeline
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
    parser = argparse.ArgumentParser(description="Run bias mitigation experiments")
    parser.add_argument("--dataset", type=str, default="gender_bias", 
                       help="Dataset to use (default: gender_bias)")
    parser.add_argument("--mode", type=str, default="both",
                       help="Mode to run (default: both)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None for random)")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top results to retrieve (default: 5)")
    parser.add_argument("--num_questions", type=int, default=2,
                       help="Number of questions to process (default: 2)")
    parser.add_argument("--project_name", type=str, default="bias-mitigation",
                       help="WandB project name (default: bias-mitigation)")
    args = parser.parse_args()
    run_experiment(args)
