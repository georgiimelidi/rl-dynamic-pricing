import numpy as np
import torch

from pricing_rl.env import DynamicPricingEnv, PricingEnvConfig
from pricing_rl.agent import DQNAgent
from pricing_rl.replay_buffer import ReplayBuffer
from pricing_rl.utils import set_seed, ensure_dir


def train():
    config = PricingEnvConfig()
    set_seed(config.seed)

    env = DynamicPricingEnv(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
    )

    replay_buffer = ReplayBuffer(capacity=10000)

    num_episodes = 600
    batch_size = 128
    training_rewards = []
    training_losses = []

    ensure_dir("outputs")
    ensure_dir("outputs/figures")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update(replay_buffer, batch_size=batch_size)

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        training_rewards.append(total_reward)
        mean_loss = float(np.mean(episode_losses)) if episode_losses else np.nan
        training_losses.append(mean_loss)

        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(training_rewards[-25:])
            avg_loss = np.nanmean(training_losses[-25:])
            print(
                f"Episode {episode + 1:3d} | "
                f"avg reward (last 25): {avg_reward:8.2f} | "
                f"avg loss (last 25): {avg_loss:10.2f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    np.save("outputs/training_rewards.npy", np.array(training_rewards, dtype=np.float32))
    np.save("outputs/training_losses.npy", np.array(training_losses, dtype=np.float32))
    torch.save(agent.q_net.state_dict(), "outputs/dqn_model.pt")

    print("Training finished.")
    print("Saved:")
    print("- outputs/training_rewards.npy")
    print("- outputs/training_losses.npy")
    print("- outputs/dqn_model.pt")


if __name__ == "__main__":
    train()