from huggingface_hub import snapshot_download
import os


if __name__ == "__main__":
    CKPT_DIR = os.environ['CKPT_DIR']

    print("Downloading Qwen/Qwen2.5-1.5B-Instruct ...")
    snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct",
                      local_dir=f"{CKPT_DIR}/models/Qwen/Qwen2.5-1.5B-Instruct/base")
