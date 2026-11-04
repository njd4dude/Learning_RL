"""Show what the CNN policy sees: its input frames and what its filters detect.

Usage:
    python visualize.py [model_path] [out_path]
    python visualize.py --untrained [out_path]

Plays one episode with a `pixelcopter-pixels` model (the newest one by
default, or with --untrained a fresh network with random filters, to compare
against) and saves a video. Each video frame shows, for that step:

- the game screen in color (what you see)
- the 4 stacked grayscale frames the network actually receives
- the 16 feature maps from conv layer 1 (11x11 each)
- the 32 feature maps from conv layer 2 (4x4 each)
- the policy's flap / no-flap probabilities

Bright cells in a feature map mean that filter found its pattern in that
part of the screen.
"""

import argparse
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")  # render to arrays, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

import config
from envs import make_env
from evaluate import latest_model_path, load_saved_model
from policy import CnnPolicy, build_policy

ACTION_NAMES = ("flap", "no flap")


def _layer_outputs(policy: CnnPolicy, obs: np.ndarray):
    """Run `obs` through the policy, keeping each conv layer's output."""
    x = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0) / 255.0
    # policy.conv is [Conv2d, ReLU, Conv2d, ReLU, Flatten].
    conv1 = policy.conv[1](policy.conv[0](x))
    conv2 = policy.conv[3](policy.conv[2](conv1))
    probs = policy(x * 255.0)  # forward() does its own /255
    return conv1[0].numpy(), conv2[0].numpy(), probs[0]


def _play_episode(env, policy: CnnPolicy) -> tuple[list[dict], float]:
    """One episode, recording everything the figure needs at each step."""
    steps = []
    obs, _info = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    with torch.no_grad():
        while not (terminated or truncated):
            conv1, conv2, probs = _layer_outputs(policy, obs)
            action = Categorical(probs).sample().item()
            steps.append(
                dict(
                    screen=env.render(),
                    frames=np.asarray(obs),
                    conv1=conv1,
                    conv2=conv2,
                    probs=probs.numpy(),
                    action=action,
                    total_reward=total_reward,
                )
            )
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
    return steps, total_reward


def _build_figure(first: dict, conv1_max: float, conv2_max: float):
    """Lay out every panel once; later frames only swap the image data."""
    # 960x576 px: both divisible by 16, so the mp4 encoder doesn't resize.
    fig = plt.figure(figsize=(12, 7.2), dpi=80, facecolor="white")
    grid = fig.add_gridspec(7, 8, hspace=0.35, wspace=0.1)

    def panel(row, col, image, title=None, **kwargs):
        ax = fig.add_subplot(grid[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8)
        return ax.imshow(image, interpolation="nearest", **kwargs)

    images = {"screen": panel(0, 0, first["screen"], "game (color)")}
    images["frames"] = [
        panel(0, 1 + i, frame, f"input t-{3 - i}" if i < 3 else "input t", cmap="gray", vmin=0, vmax=255)
        for i, frame in enumerate(first["frames"])
    ]
    images["conv1"] = [
        panel(1 + i // 8, i % 8, fmap, cmap="magma", vmin=0, vmax=conv1_max)
        for i, fmap in enumerate(first["conv1"])
    ]
    images["conv2"] = [
        panel(3 + i // 8, i % 8, fmap, cmap="magma", vmin=0, vmax=conv2_max)
        for i, fmap in enumerate(first["conv2"])
    ]
    fig.text(0.02, 0.69, "conv 1\n16 filters\n11x11", fontsize=9, va="center")
    fig.text(0.02, 0.33, "conv 2\n32 filters\n4x4", fontsize=9, va="center")
    images["status"] = fig.text(0.64, 0.86, "", fontsize=11, family="monospace", va="center")
    return fig, images


def _status_text(step: int, data: dict) -> str:
    bars = "\n".join(
        f"{name:>7} {p:5.1%} {'#' * int(round(p * 20))}"
        for name, p in zip(ACTION_NAMES, data["probs"])
    )
    return f"step {step}   reward so far {data['total_reward']:.0f}\n\n{bars}\n\nchose: {ACTION_NAMES[data['action']]}"


def _untrained_policy():
    """A freshly initialized CNN with the pixels config, as train.py builds it."""
    env_id = config.PIXELCOPTER_PIXELS_ENV_ID
    env = make_env(env_id)
    hyperparameters = {
        **config.PIXELCOPTER_PIXELS_HYPERPARAMETERS,
        "env_id": env_id,
        "state_space": list(env.observation_space.shape),
        "action_space": int(env.action_space.n),
    }
    env.close()
    return build_policy(hyperparameters, "cpu"), hyperparameters


def visualize(
    model_path: str | Path | None = None,
    out_path: str | Path | None = None,
    untrained: bool = False,
) -> Path:
    if untrained:
        policy, hyperparameters = _untrained_policy()
        name = "untrained"
    else:
        model_path = Path(model_path or latest_model_path("pixelcopter-pixels"))
        policy, hyperparameters = load_saved_model(model_path)
        if not isinstance(policy, CnnPolicy):
            raise ValueError(f"{model_path} is not a CNN model; train one with --env pixelcopter-pixels.")
        policy.cpu()
        name = model_path.parent.name
    if out_path is None:
        out_path = Path(config.VIDEO_FOLDER) / f"{name}-network.mp4"
    out_path = Path(out_path)

    env = make_env(hyperparameters["env_id"], render_mode="rgb_array")
    try:
        steps, total_reward = _play_episode(env, policy)
    finally:
        env.close()

    # One brightness scale per layer for the whole video, so a map that
    # lights up is genuinely more active, not just rescaled frame by frame.
    conv1_max = max(np.percentile([s["conv1"] for s in steps], 99), 1e-6)
    conv2_max = max(np.percentile([s["conv2"] for s in steps], 99), 1e-6)
    fig, images = _build_figure(steps[0], conv1_max, conv2_max)
    fig.suptitle("untrained (random filters)" if untrained else name, fontsize=11)

    video = []
    for i, data in enumerate(steps):
        images["screen"].set_data(data["screen"])
        for im, frame in zip(images["frames"], data["frames"]):
            im.set_data(frame)
        for im, fmap in zip(images["conv1"], data["conv1"]):
            im.set_data(fmap)
        for im, fmap in zip(images["conv2"], data["conv2"]):
            im.set_data(fmap)
        images["status"].set_text(_status_text(i, data))
        fig.canvas.draw()
        video.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Slower than the game (30 fps) so the panels are readable.
    imageio.mimsave(out_path, video, fps=10)
    print(f"Episode: {len(steps)} steps, reward {total_reward:.0f}")
    print(f"Saved network view to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("out_path", nargs="?", default=None)
    parser.add_argument("--untrained", action="store_true", help="Use a fresh random network instead of a saved model.")
    args = parser.parse_args()
    if args.untrained and args.model_path and not args.out_path:
        # With --untrained there's no model, so a lone path is the output.
        args.model_path, args.out_path = None, args.model_path
    visualize(args.model_path, args.out_path, untrained=args.untrained)
