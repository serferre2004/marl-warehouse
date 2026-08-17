import gymnasium as gym
from rware.warehouse import RewardType
import time

env = gym.make(
    "rware-tiny-2ag-v2",
    msg_bits=0,
    reward_type=RewardType.THREE_STAGE,
    request_queue_size=15
)
obs = env.reset()  # a tuple of observations

episode_length = 1000
for _ in range(episode_length):
    actions = env.action_space.sample()  # the action space can be sampled
    obs, reward, terminated, truncated, info = env.step(actions)
    env.render()  # render the environment
    reward = [reward[0].item(), reward[1].item()]  # convert reward to a list of floats
    if reward != [0, 0]: 
        print(f"Non-zero reward: {reward}")  # print the reward
    time.sleep(0.1)  # slow down the simulation for better visualization