import numpy as np

import torch  # Tensor的有关运算
import torch.optim  # 优化器
import torch.nn as nn  # 神经网络层Modules & Loss函数
import torch.autograd  # Tensor的求导方法
import torch.nn.functional as F  # 激活函数，比如relu, leaky_relu, sigmoid
import gym  # 环境

LR = 0.01
GAMMA = 0.95
EPISODE_NUM = 1000
STEP = 300


class Net(nn.Module):
    def __init__(self, STATE_NUM, ACTION_NUM):
        super(Net, self).__init__()  # 使用了nn.Modules需要调用super以进行初始化
        self.fc1 = nn.Linear(in_features=STATE_NUM, out_features=20)  # 全连接层1
        self.fc2 = nn.Linear(in_features=20, out_features=ACTION_NUM)  # 全连接层2

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class Policy(object):
    def __init__(self, env):
        self.position = 0

        self.ACTION_NUM = env.action_space.n
        self.STATE_NUM = env.observation_space.shape[0]

        self.net = Net(self.STATE_NUM, self.ACTION_NUM)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=LR)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def choose_action(self, x):  # 随机选择
        x = torch.FloatTensor(x)
        y = self.net.forward(x)

        with torch.no_grad():
            prob_weights = F.softmax(y, dim=0).data.cpu().numpy()

        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)

        return action

    def save_transition(self, state, action, reward):  # 将转移存储在记忆池中
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    # * 学习
    def learn(self):
        # 计算每一步的状态价值

        discounted_reward = np.zeros_like(self.reward_memory)
        running_add = 0

        for t in reversed(range(0, len(self.reward_memory))):
            running_add = running_add * GAMMA + self.reward_memory[t]
            discounted_reward[t] = running_add

        discounted_reward -= np.mean(discounted_reward)  # 减均值
        discounted_reward /= np.std(discounted_reward)  # 除以标准差
        discounted_reward = torch.FloatTensor(discounted_reward)

        # 前向传播
        out = self.net.forward(torch.FloatTensor(self.state_memory))
        log_prob = F.cross_entropy(input=out, target=torch.LongTensor(
            self.action_memory), reduction="none")  # 计算交叉熵

        # 反向传播
        loss = torch.mean(log_prob * discounted_reward)  # 使用最大似然的方法计算loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空记忆
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


# 获取环境并取消部分限制
env = gym.make('CartPole-v0').unwrapped
agent = Policy(env)

for episode in range(EPISODE_NUM):
    state = env.reset()
    total_reward = 0
    for step in range(STEP):
        env.render()
        action = agent.choose_action(state)  # 随机选择动作
        next_state, reward, is_done, _ = env.step(action)  # 与环境交互
        agent.save_transition(state, action,  reward)  # 存储转移
        total_reward += reward
        if is_done:
            agent.learn()
            print('episode:', episode, '|reward:', total_reward)
            break

        state = next_state  # 继续新状态下的前进

env.close()
