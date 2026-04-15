from pricing_rl.env import DynamicPricingEnv, PricingEnvConfig
from pricing_rl.agent import DQNAgent
from pricing_rl.replay_buffer import ReplayBuffer

env = DynamicPricingEnv(PricingEnvConfig())
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
buffer = ReplayBuffer()

state = env.reset()

for _ in range(100):
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    buffer.push(state, action, reward, next_state, done)
    loss = agent.update(buffer, batch_size=16)
    state = next_state

    if done:
        state = env.reset()

print("buffer size:", len(buffer))
print("epsilon:", agent.epsilon)
print("last loss:", loss)