import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

# multiprocess environment
n_cpu = 1
env = DummyVecEnv([lambda: gym.make('CartPole-v0') for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1)
print("model has been setup")
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()

while True:
    re = []
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        re += [rewards[0]]
        env.render()
    print(np.sum(re))