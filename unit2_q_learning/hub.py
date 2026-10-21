"""Push a trained Q-Learning model to the Hugging Face Hub, or pull one down.

Usage:
    python hub.py push <model_path> <repo_id>
    python hub.py pull <repo_id> <filename>
"""

import datetime
import json
import pickle
import shutil
import sys
from pathlib import Path

import gymnasium as gym
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

import config
from q_learning import evaluate_agent
from record_video import record


def _env_display_name(env_id: str, env_kwargs: dict) -> str:
    name = env_id
    if env_kwargs.get("map_name"):
        name += "-" + env_kwargs["map_name"]
    if env_kwargs.get("is_slippery") is False:
        name += "-no_slippery"
    return name


def push_to_hub(model_path: str, repo_id: str, local_repo_path: str = "hub"):
    """Evaluate, generate a model card + replay video, and upload to the Hub.

    Requires a Hugging Face account + write token. Log in once beforehand
    with `huggingface-cli login`, or call `huggingface_hub.notebook_login()`
    in a notebook.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    env = gym.make(model["env_id"], render_mode="rgb_array", **model.get("env_kwargs", {}))

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    with open(repo_local_path / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    mean_reward, std_reward = evaluate_agent(
        env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]
    )

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }
    with open(repo_local_path / "results.json", "w") as f:
        json.dump(evaluate_data, f)

    env_name = _env_display_name(model["env_id"], model.get("env_kwargs", {}))
    metadata = {"tags": [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]}
    metadata.update(
        metadata_eval_result(
            model_pretty_name=repo_id.split("/")[-1],
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )
    )

    model_card = f"""# **Q-Learning** Agent playing **{model['env_id']}**
This is a trained model of a **Q-Learning** agent playing **{model['env_id']}**.

## Usage

```python
model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

# Don't forget to check if you need to add additional attributes (is_slippery=False etc)
env = gym.make(model["env_id"], **model.get("env_kwargs", {{}}))
```
"""
    readme_path = repo_local_path / "README.md"
    readme_path.write_text(model_card, encoding="utf-8")
    metadata_save(readme_path, metadata)

    # Three episode files, plus the single replay.mp4 the Hub model card expects.
    saved = record(model_path, out_path=str(repo_local_path / "replay.mp4"))
    shutil.copy(saved[0], repo_local_path / "replay.mp4")

    api.upload_folder(repo_id=repo_id, folder_path=repo_local_path, path_in_repo=".")
    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


def pull_from_hub(repo_id: str, filename: str, n_episodes: int = 100):
    """Download a model from the Hub and report its evaluation score."""
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(downloaded, "rb") as f:
        model = pickle.load(f)

    env = gym.make(model["env_id"], **model.get("env_kwargs", {}))
    mean_reward, std_reward = evaluate_agent(
        env, model["max_steps"], n_episodes, model["qtable"], model.get("eval_seed", [])
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return model


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
        if len(sys.argv) != 4:
            print("Usage: python hub.py pull <repo_id> <filename>")
            sys.exit(1)
        pull_from_hub(sys.argv[2], sys.argv[3])
