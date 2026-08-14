import gymnasium as gym
import rware

env = gym.make(
    "rware-tiny-2ag-v2",
    msg_bits=0,
)

current = env

while True:
    print(
        type(current).__name__,
        "->",
        type(current.observation_space).__name__,
        current.observation_space,
    )

    if not hasattr(current, "env"):
        break

    current = current.env

env.close()