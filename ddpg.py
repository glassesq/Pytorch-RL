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
GAMMA = 0.95
MEMORY_CAPACITY = 1000
EPISODE_NUM = 40000
TAU = 0.02


class ActorNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNet, self).__init__()  # 使用了nn.Modules需要调用super以进行初始化
        self.fc1 = nn.Linear(in_features=input_size, out_features=16)  # 全连接层1
        self.fc2 = nn.Linear(in_features=16, out_features=output_size)  # 全连接层2

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # 限定范围
        return x


class CriticNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(CriticNet, self).__init__()  # 使用了nn.Modules需要调用super以进行初始化
        self.fc1 = nn.Linear(in_features=input_size, out_features=16)  # 全连接层1
        self.fc2 = nn.Linear(in_features=16, out_features=output_size)  # 全连接层2

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)  # 拼接两个张量
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DDPG(object):
    def __init__(self, env):
        self.position = 0

        self.STATE_NUM = env.observation_space.shape[0]
        self.ACTION_NUM = env.action_space.n

        self.actor_net = ActorNet(self.STATE_NUM, 1)
        self.actor_target_net = ActorNet(self.STATE_NUM, 1)

        self.critic_net = CriticNet(self.STATE_NUM + 1, 1)
        self.critic_target_net = CriticNet(self.STATE_NUM + 1, 1)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), lr=LR)

        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=LR)

        self.learn_step_counter = 0

        self.loss_func = nn.MSELoss()

        self.memory = np.zeros((MEMORY_CAPACITY, self.STATE_NUM * 2 + 2))

    def choose_action(self, x):  # 使用actor来选择动作
        x = torch.FloatTensor(x)
        action = self.actor_net.forward(x)
        return action

    def save_transition(self, state, action, next_state, reward):  # 将转移存储在记忆池中
        transition = np.hstack(
            (state, [action, reward], next_state))  # 将参数平铺在数组之中
        self.memory[self.position % MEMORY_CAPACITY, :] = transition
        self.position += 1

    # * 学习
    def learn(self):
        self.learn_step_counter += 1

        # 经验回放机制
        sample_index = np.random.choice(
            MEMORY_CAPACITY, BATCH_SIZE)  # 在memory中抽样: memory 必然是满的，才会进入到learn
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :self.STATE_NUM])
        batch_action = torch.Tensor(
            batch_memory[:, self.STATE_NUM:self.STATE_NUM+1])  # 对应的transition中的action
        batch_reward = torch.FloatTensor(
            batch_memory[:, self.STATE_NUM+1:self.STATE_NUM+2]).view(BATCH_SIZE, -1)  # 对应的transition中的reward
        batch_next_state = torch.FloatTensor(
            batch_memory[:, -self.STATE_NUM:])  # 对应的是最后的next_state的部分

        batch_next_action = self.actor_target_net.forward(
            batch_next_state)

        q_eval = self.critic_net.forward(batch_state, batch_action)
        q_target = batch_reward + GAMMA * self.critic_target_net(
            batch_next_state, batch_next_action)

        q_eval = torch.squeeze(q_eval)
        q_target = torch.squeeze(q_target)

        # 反向传播 Critic
        loss = self.loss_func(q_eval, q_target)  # 计算loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # 反向传播 Actor
        action = self.actor_net(batch_state)
        loss = -torch.mean(self.critic_net(batch_state, action))
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # 软更新 soft update
        def soft_update(net_target, net):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU)

        soft_update(self.actor_target_net, self.actor_net)
        soft_update(self.critic_target_net, self.critic_net)


# 获取环境并取消部分限制
env = gym.make('CartPole-v0').unwrapped
ddpg = DDPG(env)

for episode in range(EPISODE_NUM):
    state = env.reset()
    total_reward = 0
    while True:
        env.render()

        action = ddpg.choose_action(state)  # 选择action epsilon-greedy 策略
        next_state, reward, is_done, _ = env.step(
            round(action.detach().numpy()[0]))  # 与环境交互

        x, x_dot, theta, theta_dot = next_state
        reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        reward2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        reward = reward1 + reward2  # 计算reward

        ddpg.save_transition(state, action.detach().numpy()[
                             0], next_state, reward)  # 存储转移

        total_reward += reward
        if ddpg.position > MEMORY_CAPACITY:  # 仅有memory满载时才学习
            ddpg.learn()
            if is_done:
                print('episode:', episode, '|reward:', total_reward)

        if is_done:
            break

        state = next_state  # 继续新状态下的前进

env.close()
