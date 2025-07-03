import numpy as np
import torch

def compute_gae(trajectory, last_value, gamma=0.99, lam=0.95):
    rewards = [step["reward"] for step in trajectory]
    values = [step["value"] for step in trajectory] + [last_value]
    dones = [step["done"] for step in trajectory]
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns
