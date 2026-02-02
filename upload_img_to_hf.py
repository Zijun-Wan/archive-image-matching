from datasets import load_dataset
from huggingface_hub import create_repo

user_name = "nujiznaw"
dataset_name = "train_images"
repo_id = f"{user_name}/{dataset_name}"
create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)

# Build a dataset from your local folder
ds = load_dataset(
    "./image", 
    data_files={
        "stamps": "stamps/**",
        "train": "train/**",
        "train_without_filtering": "train_without_filtering/**",
    })

# (Optional) train/test split

# Push to Hub (includes image previews)
from datasets import DatasetDict
DatasetDict(ds).push_to_hub(repo_id)
