import pickle
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pricing_rl.plotting import (
    plot_training_curve,
    plot_training_loss,
)
from pricing_rl.utils import ensure_dir


BG = "#F7F7FB"
GRID = "#E3E8EF"
TEXT = "#243447"
MUTED = "#7B8794"
CARD = "#FFFFFF"

DISPLAY_NAMES = {
    "dqn": "DQN",
    "inventory_aware": "Inventory Aware",
    "decreasing_price": "Decreasing",
    "fixed_low": "Fixed Low",
    "fixed_mid": "Fixed Mid",
    "fixed_high": "Fixed High",
}

POLICY_COLORS = {
    "dqn": "#4F86F7",              # blue
    "inventory_aware": "#16B6C8",  # teal (keep)
    "decreasing_price": "#8E6CFF", # violet
    "fixed_low": "#6FCF97",        # softer green (lighter, more pastel)
    "fixed_mid": "#F6A623",        # orange
    "fixed_high": "#FF5A5F",       # coral red
}

DASHBOARD_ORDER = [
    "dqn",
    "inventory_aware",
    "decreasing_price",
    "fixed_low",
    "fixed_mid",
    "fixed_high",
]


def setup_clean_axis(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def ordered_policy_names(results):
    preferred = [name for name in DASHBOARD_ORDER if name in results]
    remaining = [name for name in results.keys() if name not in preferred]
    return preferred + remaining


def plot_summary_bar_chart(results, save_path: str):
    policy_names = ordered_policy_names(results)
    means = [results[name]["mean_revenue"] for name in policy_names]
    stds = [results[name]["std_revenue"] for name in policy_names]
    colors = [POLICY_COLORS.get(name, "#4F86F7") for name in policy_names]
    labels = [DISPLAY_NAMES.get(name, name) for name in policy_names]

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="none")
    ax.set_ylabel("Mean revenue")
    ax.set_title("Policy comparison over 50 episodes", fontsize=13, fontweight="semibold")
    ax.tick_params(axis="x", rotation=20)
    setup_clean_axis(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_single_policy_trajectory(trajectory: dict, save_prefix: str, title_prefix: str):
    t = np.arange(len(trajectory["prices"]))

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.plot(t, trajectory["prices"], marker="o", linewidth=2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"{title_prefix}: price vs time", fontsize=12, fontweight="semibold")
    setup_clean_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_price.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.plot(t, trajectory["inventory"], marker="o", linewidth=2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Inventory")
    ax.set_title(f"{title_prefix}: inventory vs time", fontsize=12, fontweight="semibold")
    setup_clean_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_inventory.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.plot(t, trajectory["cum_revenue"], marker="o", linewidth=2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative revenue")
    ax.set_title(f"{title_prefix}: cumulative revenue", fontsize=12, fontweight="semibold")
    setup_clean_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_revenue.png", dpi=160)
    plt.close(fig)


def make_policy_comparison_gif(results, save_path: str):
    policy_names = ordered_policy_names(results)
    trajectories = {name: results[name]["trajectory"] for name in policy_names}
    horizon = max(len(traj["prices"]) for traj in trajectories.values())

    max_price = max(max(traj["prices"]) for traj in trajectories.values())
    max_revenue = max(max(traj["cum_revenue"]) for traj in trajectories.values())
    max_inventory = max(max(traj["inventory"]) for traj in trajectories.values())

    frames = []

    for t_end in range(1, horizon + 1):
        fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.4), facecolor="white")

        for ax in axes:
            setup_clean_axis(ax)

        for name in policy_names:
            traj = trajectories[name]
            color = POLICY_COLORS.get(name, "#4F86F7")
            label = DISPLAY_NAMES.get(name, name)

            price_len = min(t_end, len(traj["prices"]))
            inv_len = min(t_end, len(traj["inventory"]))
            rev_len = min(t_end, len(traj["cum_revenue"]))

            # 1) Revenue
            axes[0].plot(
                np.arange(rev_len),
                traj["cum_revenue"][:rev_len],
                color=color,
                linewidth=3.0,
                alpha=0.60,
                solid_capstyle="round",
                label=label,
            )
            axes[0].scatter(
                [rev_len - 1],
                [traj["cum_revenue"][rev_len - 1]],
                s=48,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidths=1.6,
                zorder=5,
            )

            # 2) Price
            axes[1].plot(
                np.arange(price_len),
                traj["prices"][:price_len],
                color=color,
                linewidth=3.0,
                alpha=0.60,
                solid_capstyle="round",
                label=label,
            )
            axes[1].scatter(
                [price_len - 1],
                [traj["prices"][price_len - 1]],
                s=48,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidths=1.6,
                zorder=5,
            )

            # 3) Inventory
            axes[2].plot(
                np.arange(inv_len),
                traj["inventory"][:inv_len],
                color=color,
                linewidth=3.0,
                alpha=0.60,
                solid_capstyle="round",
                label=label,
            )
            axes[2].scatter(
                [inv_len - 1],
                [traj["inventory"][inv_len - 1]],
                s=48,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidths=1.6,
                zorder=5,
            )

        axes[0].set_title("Cumulative revenue vs time", fontsize=12, fontweight="semibold", color=TEXT)
        axes[0].set_ylabel("Revenue")
        axes[0].set_ylim(0, max_revenue * 1.12)

        axes[1].set_title("Price vs time", fontsize=12, fontweight="semibold", color=TEXT)
        axes[1].set_ylabel("Price")
        axes[1].set_ylim(0, max_price * 1.12)

        axes[2].set_title("Inventory vs time", fontsize=12, fontweight="semibold", color=TEXT)
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Inventory")
        axes[2].set_ylim(0, max_inventory * 1.08)

        handles = [
            plt.Line2D(
                [0], [0],
                color=POLICY_COLORS.get(name, "#4F86F7"),
                marker="o",
                markerfacecolor=POLICY_COLORS.get(name, "#4F86F7"),
                markeredgecolor="white",
                markeredgewidth=1.6,
                linewidth=3,
                alpha=0.75,
                markersize=7,
                label=DISPLAY_NAMES.get(name, name),
            )
            for name in policy_names
        ]

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        fig.suptitle(
            "Dynamic Pricing Policy Comparison",
            fontsize=15,
            fontweight="semibold",
            color=TEXT,
            y=0.985,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.88])

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, duration=0.45)


def draw_value_card(ax, x, y, w, h, title, value, accent_color, value_size=20):
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0,
        facecolor=CARD,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(card)

    ax.text(
        x + 0.04 * w,
        y + 0.72 * h,
        title,
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="center",
        ha="left",
        zorder=2,
    )
    ax.text(
        x + 0.04 * w,
        y + 0.34 * h,
        value,
        transform=ax.transAxes,
        fontsize=value_size,
        fontweight="bold",
        color=accent_color,
        va="center",
        ha="left",
        zorder=2,
    )


def make_dashboard_gif(results, save_path: str):
    policy_names = ordered_policy_names(results)
    trajectories = {name: results[name]["trajectory"] for name in policy_names}
    horizon = max(len(traj["prices"]) for traj in trajectories.values())
    initial_inventory = max(
        max(traj["inventory"]) if len(traj["inventory"]) > 0 else 0
        for traj in trajectories.values()
    )

    frames = []

    for t_end in range(1, horizon + 1):
        fig, axes = plt.subplots(len(policy_names), 1, figsize=(10, 2.45 * len(policy_names)), facecolor=BG)
        if len(policy_names) == 1:
            axes = [axes]

        for ax, name in zip(axes, policy_names):
            traj = trajectories[name]
            idx = min(t_end - 1, len(traj["prices"]) - 1)

            current_price = traj["prices"][idx]
            current_revenue = traj["cum_revenue"][idx]
            current_inventory = traj["inventory"][idx]
            current_time = min(t_end, len(traj["prices"]))
            total_time = len(traj["prices"])

            accent = POLICY_COLORS.get(name, "#4F86F7")
            display_name = DISPLAY_NAMES.get(name, name)

            ax.clear()
            ax.set_facecolor(BG)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            row_card = FancyBboxPatch(
                (0.01, 0.06), 0.98, 0.88,
                boxstyle="round,pad=0.018,rounding_size=0.03",
                linewidth=0,
                facecolor="#EEF2F8",
                transform=ax.transAxes,
                zorder=0,
            )
            ax.add_patch(row_card)

            ax.text(
                0.03, 0.84, display_name,
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                color=TEXT,
                ha="left",
                va="center",
            )
            ax.text(
                0.28, 0.84, f"Step {current_time}/{total_time}",
                transform=ax.transAxes,
                fontsize=10,
                color=MUTED,
                ha="left",
                va="center",
            )

            draw_value_card(ax, 0.03, 0.22, 0.18, 0.42, "Price", f"{current_price:.2f}", accent, value_size=20)
            draw_value_card(ax, 0.24, 0.22, 0.24, 0.42, "Revenue", f"{current_revenue:.0f}", accent, value_size=20)

            ax.text(
                0.53, 0.58, "Inventory",
                transform=ax.transAxes,
                fontsize=9,
                color=MUTED,
                ha="left",
                va="center",
            )
            ax.text(
                0.53, 0.40, f"{current_inventory}/{initial_inventory}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="semibold",
                color=TEXT,
                ha="left",
                va="center",
            )

            bar_x, bar_y, bar_w, bar_h = 0.67, 0.34, 0.28, 0.14
            ax.add_patch(
                FancyBboxPatch(
                    (bar_x, bar_y), bar_w, bar_h,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    linewidth=0,
                    facecolor="#DCE3EC",
                    transform=ax.transAxes,
                    zorder=1,
                )
            )

            fill_ratio = current_inventory / initial_inventory if initial_inventory > 0 else 0.0
            ax.add_patch(
                FancyBboxPatch(
                    (bar_x, bar_y), bar_w * fill_ratio, bar_h,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    linewidth=0,
                    facecolor="#A7B3C2",
                    transform=ax.transAxes,
                    zorder=2,
                )
            )

        fig.suptitle("Policy Dashboard", fontsize=15, fontweight="semibold", color=TEXT, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, duration=0.45)


def main():
    ensure_dir("outputs/figures")

    training_rewards = np.load("outputs/training_rewards.npy")
    training_losses = np.load("outputs/training_losses.npy")

    plot_training_curve(training_rewards, "outputs/figures/training_rewards.png")
    plot_training_loss(training_losses, "outputs/figures/training_losses.png")

    with open("outputs/eval_results.pkl", "rb") as f:
        results = pickle.load(f)

    plot_summary_bar_chart(results, "outputs/figures/policy_summary.png")

    for policy_name, result in results.items():
        plot_single_policy_trajectory(
            result["trajectory"],
            save_prefix=f"outputs/figures/{policy_name}",
            title_prefix=DISPLAY_NAMES.get(policy_name, policy_name),
        )

    make_policy_comparison_gif(results, "outputs/figures/policy_comparison.gif")
    make_dashboard_gif(results, "outputs/figures/policy_dashboard.gif")

    print("Saved plots to outputs/figures/")


if __name__ == "__main__":
    main()