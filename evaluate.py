import pickle
import numpy as np
import torch

from pricing_rl.env import DynamicPricingEnv, PricingEnvConfig
from pricing_rl.agent import DQNAgent
from pricing_rl.baselines import (
    FixedPricePolicy,
    DecreasingPricePolicy,
    InventoryAwarePolicy,
)
from pricing_rl.utils import set_seed, ensure_dir


def rollout_policy(env, policy, greedy: bool = True):
    state = env.reset()
    done = False

    prices = []
    inventory = [env.inventory]
    rewards = []
    cum_revenue = []
    arrivals = []
    conversions = []
    sales = []

    total_revenue = 0.0

    while not done:
        if policy.__class__.__name__ == "DQNAgent":
            action = policy.select_action(state, greedy=greedy)
        else:
            action = policy.select_action(state)

        next_state, reward, done, info = env.step(action)

        prices.append(info["price"])
        arrivals.append(info["arrivals"])
        conversions.append(info["conversion_prob"])
        sales.append(info["units_sold"])

        rewards.append(reward)
        total_revenue += reward
        cum_revenue.append(total_revenue)
        inventory.append(info["inventory"])

        state = next_state

    return {
        "total_revenue": total_revenue,
        "inventory_used": env.config.inventory - env.inventory,
        "prices": prices,
        "inventory": inventory[:-1],
        "cum_revenue": cum_revenue,
        "step_rewards": rewards,
        "arrivals": arrivals,
        "conversion_prob": conversions,
        "sales": sales,
    }


def build_policies(config, env):
    dqn = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device="cpu",
    )
    dqn.q_net.load_state_dict(torch.load("outputs/dqn_model.pt", map_location="cpu"))
    dqn.target_net.load_state_dict(dqn.q_net.state_dict())
    dqn.epsilon = 0.0

    return {
        "dqn": dqn,
        "fixed_low": FixedPricePolicy(action=0),
        "fixed_mid": FixedPricePolicy(action=len(config.prices) // 2),
        "fixed_high": FixedPricePolicy(action=len(config.prices) - 1),
        "decreasing_price": DecreasingPricePolicy(num_actions=len(config.prices)),
        "inventory_aware": InventoryAwarePolicy(num_actions=len(config.prices)),
    }


def evaluate_policy_over_many_episodes(policy, config, num_episodes: int = 50):
    revenues = []
    inventory_used = []
    representative_trajectory = None

    for episode_idx in range(num_episodes):
        episode_config = PricingEnvConfig(
            inventory=config.inventory,
            horizon=config.horizon,
            prices=config.prices,
            base_arrival_rate=config.base_arrival_rate,
            time_pattern=config.time_pattern,
            conversion_ref_price=config.conversion_ref_price,
            conversion_slope=config.conversion_slope,
            seed=config.seed + episode_idx,
        )

        env = DynamicPricingEnv(episode_config)
        trajectory = rollout_policy(env, policy, greedy=True)

        revenues.append(trajectory["total_revenue"])
        inventory_used.append(trajectory["inventory_used"])

        if episode_idx == 0:
            representative_trajectory = trajectory

    return {
        "mean_revenue": float(np.mean(revenues)),
        "std_revenue": float(np.std(revenues)),
        "mean_inventory_used": float(np.mean(inventory_used)),
        "std_inventory_used": float(np.std(inventory_used)),
        "revenues": revenues,
        "inventory_used_values": inventory_used,
        "trajectory": representative_trajectory,
    }


def evaluate():
    config = PricingEnvConfig()
    set_seed(config.seed)
    ensure_dir("outputs")

    env = DynamicPricingEnv(config)
    policies = build_policies(config, env)

    results = {}

    for name, policy in policies.items():
        results[name] = evaluate_policy_over_many_episodes(
            policy=policy,
            config=config,
            num_episodes=50,
        )

    with open("outputs/eval_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nEvaluation summary over 50 episodes:")
    for name, res in results.items():
        print(
            f"{name:18s} | "
            f"mean revenue = {res['mean_revenue']:8.2f} | "
            f"std = {res['std_revenue']:7.2f} | "
            f"mean inventory used = {res['mean_inventory_used']:6.2f}"
        )

    print("\nSaved:")
    print("- outputs/eval_results.pkl")


if __name__ == "__main__":
    evaluate()