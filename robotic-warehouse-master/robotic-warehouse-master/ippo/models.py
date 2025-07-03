import torch
import torch.nn as nn

'''
Actor-Critic架构共享特征提取网络，并使用独立的下游网络进行动作决策与价值估计
'''

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim_1=128, hidden_dim_2=64):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, act_dim),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1))

    def actor_forward(self, x):
        x = self.shared_net(x)
        x = self.actor(x)
        return x

    def critic_forward(self, x):
        x = self.shared_net(x)
        x = self.critic(x)
        return x


