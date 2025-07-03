import numpy as np

class MultiAgentRwareEnv:
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs, _ = self.env.reset()
        return [obs[i] for i in range(self.n_agents)]

    def step(self, actions):
        obs, rewards, done, truncated, info = self.env.step(actions)
        obs = [obs[i] for i in range(self.n_agents)]
        rewards = [rewards[i] for i in range(self.n_agents)]
        return obs, rewards, done, truncated, info

    def render(self):
        self.env.render()
