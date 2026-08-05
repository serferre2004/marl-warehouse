import gymnasium as gym
import numpy as np
import rware
import time
env = gym.make("rware-tiny-3ag-v2")
obs = env.reset()  # a tuple of observations

A=['Noop', 'Forward', 'Turn Left', 'Turn Right', 'Load/Unload']
episode_length = 100
for _ in range(episode_length):
    actions = [np.random.choice(len(A), p=[0, 0.6, 0.2, 0.2, 0]) for _ in range(3)]  # the action space can be sampled
    print(list(map(lambda x: A[x], actions)))  # (1, 0)
    obs, reward, terminated, truncated, info = env.step(actions)
    env.render()  # render the environment
    print(info["episode_collisions"])  # number of collisions in the episode
    time.sleep(3)  # slow down the simulation for better visualization  