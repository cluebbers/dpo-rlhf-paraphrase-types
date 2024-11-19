import subprocess
import argparse

from huggingface_hub import login

def run_lm_eval(pretrained, output_path):
    command = [
        "lm-eval",
        f"--model_args",
        f"pretrained={pretrained},dtype=bfloat16",
        "--tasks=leaderboard",
        "--batch_size=auto",
        f"--output_path={output_path}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm-eval with specified model and output path")
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results")

    args = parser.parse_args()
    login(new_session=False)
    run_lm_eval(args.pretrained, args.output_path)