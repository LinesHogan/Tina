import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, login

def upload_model_to_hf(local_path, hf_token, repo_name, repo_id=None):
    """
    上传本地模型到Hugging Face Hub
    
    Args:
        local_path: 本地模型路径
        hf_token: Hugging Face API token
        repo_name: 仓库名称
        repo_id: 完整的仓库ID (username/repo_name)，如果不提供则使用当前用户名
    """
    # 登录HF
    login(token=hf_token)
    
    # 初始化API
    api = HfApi()
    
    # 如果没有提供repo_id，使用当前用户名
    if repo_id is None:
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{repo_name}"
    
    print(f"Loading model from {local_path}")
    
    # 加载tokenizer和model
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(local_path)
        
        print(f"Creating repository: {repo_id}")
        
        # 创建仓库
        api.create_repo(repo_id, exist_ok=True)
        
        print("Uploading model...")
        
        # 上传模型和tokenizer
        model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)
        
        print(f"Successfully uploaded model to https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Upload local LLM checkpoint to Hugging Face")
    parser.add_argument("--local_path", required=True, help="Path to local model checkpoint")
    parser.add_argument("--hf_token", required=True, help="Hugging Face API token")
    parser.add_argument("--repo_name", required=True, help="Repository name on HF")
    parser.add_argument("--repo_id", help="Full repository ID (username/repo_name)")
    
    args = parser.parse_args()
    
    upload_model_to_hf(
        local_path=args.local_path,
        hf_token=args.hf_token,
        repo_name=args.repo_name,
        repo_id=args.repo_id
    )

if __name__ == "__main__":
    main()