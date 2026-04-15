from pricing_rl.env import DynamicPricingEnv, PricingEnvConfig

env = DynamicPricingEnv(PricingEnvConfig())
state = env.reset()
print("initial state:", state)

for _ in range(3):
    next_state, reward, done, info = env.step(2)
    print(next_state, reward, done, info)