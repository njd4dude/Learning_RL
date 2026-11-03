"""Push a trained REINFORCE model to the Hugging Face Hub, or pull one down.

Usage:
    python hub.py push <model_path> <repo_id>
    python hub.py pull <repo_id>
"""

import __main__
import datetime
import json
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

import config
from envs import make_env
from evaluate import load_saved_model, n_evaluation_episodes
from policy import Policy
from record_video import record_episode
from reinforce import evaluate_agent


def push_to_hub(model_path: str, repo_id: str):
    """Evaluate, generate a model card + replay video, and upload to the Hub.

    Uploads the same files as the course notebook (model.pt as a whole pickled
    module, hyperparameters.json, results.json, README.md, replay.mp4), so the
    course leaderboard can score it. Log in once beforehand with
    `huggingface-cli login`.
    """
    policy, hyperparameters = load_saved_model(model_path)
    env_id = hyperparameters["env_id"]
    # Score with the current config.py value; record it so the Hub files agree.
    hyperparameters = {**hyperparameters, "n_evaluation_episodes": n_evaluation_episodes(env_id)}
    repo_name = repo_id.split("/")[-1]

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        torch.save(policy, local_directory / "model.pt")
        with open(local_directory / "hyperparameters.json", "w") as f:
            json.dump(hyperparameters, f)

        eval_env = make_env(env_id)
        mean_reward, std_reward = evaluate_agent(
            eval_env, hyperparameters["max_t"], hyperparameters["n_evaluation_episodes"], policy
        )
        eval_env.close()

        evaluate_data = {
            "env_id": env_id,
            "mean_reward": mean_reward,
            "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
            "eval_datetime": datetime.datetime.now().isoformat(),
        }
        with open(local_directory / "results.json", "w") as f:
            json.dump(evaluate_data, f)

        metadata = {
            "tags": [
                env_id,
                "reinforce",
                "reinforcement-learning",
                "custom-implementation",
                "deep-rl-class",
            ]
        }
        metadata.update(
            metadata_eval_result(
                model_pretty_name=repo_name,
                task_pretty_name="reinforcement-learning",
                task_id="reinforcement-learning",
                metrics_pretty_name="mean_reward",
                metrics_id="mean_reward",
                metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
                dataset_pretty_name=env_id,
                dataset_id=env_id,
            )
        )

        model_card = f"""# **Reinforce** Agent playing **{env_id}**
This is a trained model of a **Reinforce** agent playing **{env_id}**.
To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
"""
        readme_path = local_directory / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")
        metadata_save(readme_path, metadata)

        video_env = make_env(env_id, render_mode="rgb_array")
        try:
            record_episode(video_env, policy, local_directory / "replay.mp4")
        finally:
            video_env.close()

        api.upload_folder(repo_id=repo_id, folder_path=local_directory, path_in_repo=".")

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


def pull_from_hub(repo_id: str):
    """Download a course-format model from the Hub and report its score.

    model.pt is a pickled module, and unpickling runs code, so only pull
    repos you trust. Models pushed from the course notebook were pickled as
    `__main__.Policy`; pointing that name at our Policy (same layer names)
    lets them load here.
    """
    if not hasattr(__main__, "Policy"):
        __main__.Policy = Policy

    model_file = hf_hub_download(repo_id=repo_id, filename="model.pt")
    hyperparameters_file = hf_hub_download(repo_id=repo_id, filename="hyperparameters.json")
    policy = torch.load(model_file, map_location=config.DEVICE, weights_only=False)
    with open(hyperparameters_file) as f:
        hyperparameters = json.load(f)

    env = make_env(hyperparameters["env_id"])
    mean_reward, std_reward = evaluate_agent(
        env, hyperparameters["max_t"], hyperparameters["n_evaluation_episodes"], policy
    )
    env.close()
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return policy, hyperparameters


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "pull"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "push":
        if len(sys.argv) != 4:
            print("Usage: python hub.py push <model_path> <repo_id>")
            sys.exit(1)
        push_to_hub(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) != 3:
            print("Usage: python hub.py pull <repo_id>")
            sys.exit(1)
        pull_from_hub(sys.argv[2])
