from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_training_curve(rewards, save_path: str) -> None:
    ensure_parent_dir(save_path)

    plt.figure(figsize=(7, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training reward per episode")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_loss(losses, save_path: str) -> None:
    ensure_parent_dir(save_path)

    losses = np.array(losses, dtype=float)
    valid_mask = ~np.isnan(losses)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(losses))[valid_mask], losses[valid_mask])
    plt.xlabel("Episode")
    plt.ylabel("Average loss")
    plt.title("Training loss per episode")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_policy_trajectory(trajectory: dict, save_prefix: str, title_prefix: str) -> None:
    ensure_parent_dir(save_prefix + "_dummy.png")

    t = np.arange(len(trajectory["prices"]))

    plt.figure(figsize=(7, 4))
    plt.plot(t, trajectory["prices"], marker="o")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"{title_prefix}: price vs time")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_price.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, trajectory["inventory"], marker="o")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title(f"{title_prefix}: inventory vs time")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_inventory.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, trajectory["cum_revenue"], marker="o")
    plt.xlabel("Time")
    plt.ylabel("Cumulative revenue")
    plt.title(f"{title_prefix}: cumulative revenue")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_revenue.png")
    plt.close()