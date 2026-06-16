import os
from huggingface_hub import login, HfApi
from datasets import load_dataset

def push_dataset():
    print("🚀 Preparing to push dataset to Hugging Face Hub...")
    
    # 1. Login check
    # We'll try to use the stored token first
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Logged in as: {username}")
    except Exception:
        print("⚠️  Not logged in or token invalid.")
        token = input("Please enter your Hugging Face Write Token: ").strip()
        login(token=token)
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']

    # 2. Load the dataset
    dataset_path = "drejja_perfect_instruct_inflated.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: {dataset_path} not found!")
        return

    print(f"📂 Loading {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"✅ Dataset loaded: {len(dataset)} rows")

    # 3. Ask for repo name
    default_repo_idx = "drejja-perfect-instruct"
    repo_name_input = input(f"Enter repository name (default: {username}/{default_repo_idx}): ").strip()
    
    if not repo_name_input:
        repo_id = f"{username}/{default_repo_idx}"
    elif "/" not in repo_name_input:
        repo_id = f"{username}/{repo_name_input}"
    else:
        repo_id = repo_name_input

    # 4. Push to hub
    print(f"📤 Pushing to {repo_id}...")
    try:
        dataset.push_to_hub(repo_id, private=False)
        print(f"🎉 Success! Dataset available at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")

if __name__ == "__main__":
    push_dataset()
