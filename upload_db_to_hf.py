from huggingface_hub import create_repo, upload_folder

user_name = "nujiznaw"
dataset_name = "vocab_db"
repo_id = f"{user_name}/{dataset_name}"
create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="vocab_db/your_db_name",  # local folder with the 3 .pkl.gz files
    path_in_repo="your_db_name", # optional subdir name on the Hub
)

print("Pushed to:", repo_id)
