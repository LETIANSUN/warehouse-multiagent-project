import torch
import torch.optim as optim
from utils import compute_gae

class IPPOAgent:
    def __init__(self, obs_dim, act_dim, device):
        from models import MLPPolicy, MLPValue
        self.policy = MLPPolicy(obs_dim, act_dim).to(device)
        self.value = MLPValue(obs_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)


def ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()



def train_ippo(env, agents, config):
    device = config.get("device", "cpu")
    rollout_length = config.get("rollout_length", 128)
    epochs = config.get("epochs", 4)
    mini_batch_size = config.get("mini_batch_size", 32)
    gamma = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)
    clip_eps = config.get("clip_eps", 0.2)
    value_coef = config.get("value_coef", 0.5)
    entropy_coef = config.get("entropy_coef", 0.01)
    max_train_steps = config.get("max_train_steps", 10000)

    for step in range(max_train_steps):
        # 1. 采集数据
        trajectories = collect_rollout(env, agents, rollout_length, device)

        # 2. 每个 agent 独立更新
        for i, agent in enumerate(agents):
            # 2.1 整理数据
            obs = torch.tensor([t["obs"] for t in trajectories[i]], dtype=torch.float32, device=device)
            actions = torch.tensor([t["action"] for t in trajectories[i]], dtype=torch.long, device=device)
            old_log_probs = torch.tensor([t["log_prob"] for t in trajectories[i]], dtype=torch.float32, device=device)
            rewards = [t["reward"] for t in trajectories[i]]
            dones = [t["done"] for t in trajectories[i]]

            # 2.2 计算最后一个状态的 value
            with torch.no_grad():
                last_obs = torch.tensor(trajectories[i][-1]["next_obs"], dtype=torch.float32, device=device).unsqueeze(0)
                last_value = agent.value(last_obs).item()

            # 2.3 计算 advantage 和 returns
            advantages, returns = compute_gae(trajectories[i], last_value, gamma, lam)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            # 2.4 多轮 epoch 随机小批量更新
            n = len(obs)
            for _ in range(epochs):
                idx = torch.randperm(n)
                for start in range(0, n, mini_batch_size):
                    end = start + mini_batch_size
                    mb_idx = idx[start:end]
                    mb_obs = obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_returns = returns[mb_idx]

                    # 策略网络
                    logits = agent.policy(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    # PPO 损失
                    policy_loss = ppo_loss(new_log_probs, mb_old_log_probs, mb_advantages, clip_eps)
                    # 价值损失
                    values = agent.value(mb_obs).squeeze()
                    value_loss = F.mse_loss(values, mb_returns)
                    # 总损失
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                    # 优化
                    agent.policy_optimizer.zero_grad()
                    agent.value_optimizer.zero_grad()
                    loss.backward()
                    agent.policy_optimizer.step()
                    agent.value_optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: agent0 reward sum = {sum([t['reward'] for t in trajectories[0]])}")