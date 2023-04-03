import collections
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """
    经验回放区：
    保存 (state, action, reward, next_state, done) 的采样序列
    暗含关于环境的信息，p(r|s, a) p(s_|s, a)，即当前状态 s 做动作 a 后，环境会给多大奖励 r，以及转移到何种下一状态 s_
    """

    def __init__(self, buffer_capacity):
        # 队列最多保存 buffer_capacity 个元素
        self.buffer = collections.deque(maxlen=buffer_capacity)

    @property
    def size(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # 保存一条经验 ，即 (state, action, reward, done, next_state)
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        # 先用 * 解压 sample 列表，得到一批独立样本，之后按属性打包
        batch_s, batch_a, batch_r, batch_s_, batch_done = zip(*samples)

        # 返回 array，之后转为 tensor 更快
        return np.array(batch_s), np.array(batch_a), np.array(batch_r), np.array(batch_s_), np.array(batch_done)


class Actor(nn.Module):
    """
    定义 Actor 策略网络结构
    """

    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        """
        :param state_dim: state 转为 1 维的维度
        :param action_dim: action 转为 1 维的维度
        :param action_bound:
        """
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.action_bound = torch.tensor(action_bound)

    def forward(self, s):
        a = self.net(s) * self.action_bound
        return a


class Critic(nn.Module):
    """
    定义 Critic 评估网络结构
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        :param state_dim: state 转为 1 维的维度
        :param action_dim: action 转为 1 维的维度
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, a):
        s_a = torch.cat((s, a), dim=1)
        q = self.net(s_a)
        return q


class DDPG:
    def __init__(self, state_dim, action_dim, actor_hidden_dim, critic_hidden_dim, action_bound, actor_lr, critic_lr,
                 gamma, sigma, tau, device):
        self.device = device
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = Actor(state_dim, action_dim, actor_hidden_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.tar_actor = Actor(state_dim, action_dim, actor_hidden_dim, action_bound).to(device)
        self.tar_critic = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.tar_actor.load_state_dict(self.actor.state_dict())
        self.tar_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma

        # 为了使DDPG策略更好地进行探索，在训练时对其行为增加了干扰，原始DDPG论文的作者建议使用时间相关的 OU噪声
        # 但最近的结果表明，不相关的均值零高斯噪声效果很好，由于后者更简单，因此是首选
        self.sigma = sigma

        # 目标网络软更新参数
        self.tau = tau

    def choose_action(self, state, epsilon=1):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + epsilon * np.random.normal(0, self.sigma, self.action_dim)
        # action = (nn.Tanh()(torch.from_numpy(action)) * self.action_bound).numpy()
        return action

    def update(self, buffer: ReplayBuffer, batch_size):
        if buffer.size < batch_size:
            return

        batch_s, batch_a, batch_r, batch_s_, batch_done = buffer.sample(batch_size)

        # 转变为 tensor 计算梯度
        batch_s = torch.tensor(batch_s, dtype=torch.float).to(self.device)
        batch_a = torch.tensor(batch_a, dtype=torch.int).reshape(-1, 1).to(self.device)
        batch_r = torch.tensor(batch_r, dtype=torch.float).reshape(-1, 1).to(self.device)
        batch_s_ = torch.tensor(batch_s_, dtype=torch.float).to(self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.int).reshape(-1, 1).to(self.device)

        # critic 评估误差
        q_targets = batch_r + (1 - batch_done) * self.gamma * self.tar_critic(batch_s_, self.tar_actor(batch_s_))
        critic_loss = F.mse_loss(self.critic(batch_s, batch_a), q_targets)

        # 梯度下降
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 在反向传播算出梯度之后，参数更新之前，做这一步防止梯度爆炸或梯度消失
        # nn.utils.clip_grad_value_(self.critic.parameters(), 100)
        self.critic_optimizer.step()

        # 定义策略评估函数（目标函数） J = E[Q(S,A)]，使用梯度上升更新策略
        actor_loss = -torch.mean(self.critic(batch_s, self.actor(batch_s)))

        # 梯度下降
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 在反向传播算出梯度之后，参数更新之前，做这一步防止梯度爆炸或梯度消失
        # nn.utils.clip_grad_value_(self.actor.parameters(), 100)
        self.actor_optimizer.step()

        # soft update 更新
        self.soft_update(self.actor, self.tar_actor)
        self.soft_update(self.critic, self.tar_critic)

    def soft_update(self, source, target):
        for target_params, source_params in zip(target.parameters(), source.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.tau) + source_params.data * self.tau)

    def save_net(self, sav_dir='', sav_struc=False):
        if sav_struc:
            # 保存模型的结构和参数
            torch.save(self.tar_actor, sav_dir + '/' + 'tar_actor.pkl')
            torch.save(self.tar_critic, sav_dir + '/' + 'tar_critic.pkl')
            torch.save(self.actor, sav_dir + '/' + 'actor.pkl')
            torch.save(self.critic, sav_dir + '/' + 'critic.pkl')
        else:
            # 只保存模型的参数
            torch.save(self.tar_actor.state_dict(), sav_dir + '/' + 'tar_actor.pkl')
            torch.save(self.tar_critic.state_dict(), sav_dir + '/' + 'tar_critic.pkl')
            torch.save(self.actor.state_dict(), sav_dir + '/' + 'actor.pkl')
            torch.save(self.critic.state_dict(), sav_dir + '/' + 'critic.pkl')
