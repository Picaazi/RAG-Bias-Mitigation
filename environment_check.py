import sys
import os
import importlib
import openai

def check_package_installed(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is installed")
        return True
    except ImportError:
        print(f"{package_name} is NOT installed")
        return False

def check_openai_key():
    """Check if OpenAI API key is set and working"""
    try:
        
        # Check if API key is set
        api_key = os.getenv('OPENAI_KEY')
        if not api_key:
            print("OPENAI_KEY environment variable is not set")
            return False
        
        print("OPENAI_KEY environment variable is set")
        
        # Test API key by making a simple request
        client = openai.OpenAI(api_key=api_key)
        response = client.models.list()
        print("OpenAI API key is working")
        return True
        
    except ImportError:
        print("openai package is not installed")
        return False
    except Exception as e:
        print(f"OpenAI API key test failed: {e}")
        return False

def main():
    print("Environment Check")
    print("=" * 30)
    
    # Common packages for RAG systems
    packages = [
        'openai', 'numpy', 'pandas', 'transformers', 'torch', 'sklearn',
        'sentence_transformers', 'FlagEmbedding', 'faiss', 'rank_bm25', 
        'bm25s', 'rankify', 'datasets', 'evaluate', 'ir_datasets', 
        'langchain', 'dotenv', 'requests', 'tqdm', 'wandb', 'matplotlib', 
        'seaborn', 'accelerate', 'sentencepiece'
    ]
    
    all_packages_installed = True
    for package in packages:
        if not check_package_installed(package):
            all_packages_installed = False
    
    print("\nOpenAI API Check")
    print("-" * 20)
    openai_working = check_openai_key()
    
    print("\nSummary")
    print("-" * 10)
    if all_packages_installed and openai_working:
        print("Environment is ready!")
    else:
        print("Environment setup incomplete")

if __name__ == "__main__":
    main()