import torch
from env_wrappers import MultiAgentRwareEnv
from trainer import IPPOAgent, train_ippo
from rware.warehouse import Warehouse, RewardType

def make_env():
    env = Warehouse(
        shelf_columns=3,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=3000,
        max_steps=2000,
        reward_type=RewardType.GLOBAL,
    )
    return MultiAgentRwareEnv(env)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    print(env.observation_space)
    obs_dim = 
    act_dim = env.action_space.n
    agents = [IPPOAgent(obs_dim, act_dim, device) for _ in range(env.n_agents)]
    config = {
        "device": device,
        "rollout_length": 128,
        "epochs": 4,
        "mini_batch_size": 32,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_train_steps": 10000,
    }
    train_ippo(env, agents, config)

if __name__ == "__main__":
    main()
