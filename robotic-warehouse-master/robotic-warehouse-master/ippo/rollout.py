import torch

def collect_rollout(env, agents, rollout_length, device):
    obs = env.reset()
    n_agents = len(agents)
    trajectories = [[] for _ in range(n_agents)]

    for _ in range(rollout_length):
        actions, log_probs, values = [], [], []
        for i in range(n_agents):
            obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
            logits = agents[i].policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = agents[i].value(obs_tensor)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
        next_obs, rewards, done, truncated, info = env.step(actions)
        for i in range(n_agents):
            trajectories[i].append({
                "obs": obs[i],
                "action": actions[i],
                "reward": rewards[i],
                "log_prob": log_probs[i],
                "value": values[i],
                "done": done,
                "next_obs": next_obs[i]
            })
        obs = next_obs
        if done:
            break
    return trajectories
