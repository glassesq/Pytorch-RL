import numpy as np

import torch  # Tensor的有关运算
import torch.optim
import torch.nn as nn  # 神经网络层Modules & Loss函数
import torch.autograd  # Tensor的求导方法
import torch.nn.functional as F  # 激活函数，比如relu, leaky_relu, sigmoid
import gym  # 环境

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
EPISODE_NUM = 400


class Net(nn.Module):
    def __init__(self, STATE_NUM, ACTION_NUM):
        super(Net, self).__init__()  # 使用了nn.Modules需要调用super以进行初始化
        self.fc1 = nn.Linear(in_features=STATE_NUM, out_features=10)  # 输入层
        self.fc2 = nn.Linear(in_features=10, out_features=ACTION_NUM)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.fc2(x)
        return action_value


class DDQN(object):
    def __init__(self, env):
        self.position = 0

        self.ACTION_NUM = env.action_space.n
        self.STATE_NUM = env.observation_space.shape[0]
        self.ENV_A_SHAPE = 0 if isinstance(
            env.action_space.sample(), int) else env.action_space.sample().shape

        self.eval_net = Net(self.STATE_NUM, self.ACTION_NUM)
        self.target_net = Net(self.STATE_NUM, self.ACTION_NUM)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.learn_step_counter = 0

        self.loss_func = nn.MSELoss()

        self.memory = np.zeros((MEMORY_CAPACITY, self.STATE_NUM * 2 + 2))

    def choose_action(self, x):  # epsilon-greedy策略：避免收敛在局部最优
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() <= EPSILON:  # 以epsilon的概率利用
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            if self.ENV_A_SHAPE == 0:
                action = action[0]
            else:
                action = action.reshape(self.ENV_A_SHAPE)
        else:  # 以1-epsilon的概率探索
            action = np.random.randint(0, self.ACTION_NUM)
            if self.ENV_A_SHAPE == 0:
                action = action
            else:
                action = action.reshape(self.ENV_A_SHAPE)
        return action

    def save_transition(self, state, action, next_state, reward):  # 将转移存储在记忆池中
        transition = np.hstack(
            (state, [action, reward], next_state))  # 将参数平铺在数组之中
        self.memory[self.position % MEMORY_CAPACITY, :] = transition
        self.position += 1

    # * 学习
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # TargetNet: 用eval_net来更改targetnet的参数
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(
            MEMORY_CAPACITY, BATCH_SIZE)  # 在memory中抽样: memory 必然是满的，才会进入到learn
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :self.STATE_NUM])
        batch_action = torch.LongTensor(
            batch_memory[:, self.STATE_NUM:self.STATE_NUM+1].astype(int))  # 对应的transition中的action
        batch_reward = torch.FloatTensor(
            batch_memory[:, self.STATE_NUM+1:self.STATE_NUM+2])  # 对应的transition中的reward
        batch_next_state = torch.FloatTensor(
            batch_memory[:, -self.STATE_NUM:])  # 对应的是最后的next_state的部分

        q_eval = self.eval_net(batch_state).gather(
            1, batch_action)  # 从eval中获取价值函数
        q_next = self.target_net(batch_next_state).detach()  # 切一段下来，避免反向传播
        q_target = batch_reward + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)  # 使用target_net来推荐最大reward值

        loss = self.loss_func(q_eval, q_target)  # 计算loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 获取环境并取消部分限制
env = gym.make('CartPole-v1').unwrapped
ddqn = DDQN(env)

for episode in range(EPISODE_NUM):
    state = env.reset()
    total_reward = 0
    while True:
        env.render()

        action = ddqn.choose_action(state)  # 选择action epsilon-greedy 策略
        next_state, reward, is_done, _ = env.step(action)  # 与环境交互

        x, x_dot, theta, theta_dot = next_state
        reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        reward2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        reward = reward1 + reward2  # 计算reward

        ddqn.save_transition(state, action, next_state, reward)  # 存储转移

        total_reward += reward
        if ddqn.position > MEMORY_CAPACITY:  # 仅有memory满载时才学习
            ddqn.learn()
            if is_done:  # CartPole-v0游戏在坚持200步后，或者杆子掉下来时会结束，此时is_done=True
                print('episode:', episode, '|reward:', total_reward)

        if is_done:
            break

        state = next_state  # 继续新状态下的前进

env.close()
