from rware.warehouse import Warehouse, RewardType
from rware.rendering import Viewer  # 如果不是 rendering，你试试 rware.utils.rendering

# 创建环境
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

# 初始化渲染窗口
viewer = Viewer(env.grid_size)
obs, _ = env.reset()

# 每一步执行一个随机动作并渲染
done = False
while not done:
    viewer.render(env)
    action = env.action_space.sample()
    print(action)  # 随机动作
    obs, reward, done, truncated, info = env.step(action)
