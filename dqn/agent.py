import collections
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:
    """
    经验回放区
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


class Net(nn.Module):
    """
    使用神经网络作函数，做价值函数逼近
    定义网络结构，即目标网络（target net）和行为网络（eval net）的网络结构
    """

    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        # 只定义两层隐藏层网络，包含 128 隐藏节点，并使用 relu 激活函数
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, s):
        # 使用神经网络，输入 s 状态，计算得到此状态 s 的所有 q(s, a)
        q_values = self.net(s)
        return q_values


class DQN:
    """
    使用 DQN 算法，利用经验回放区的数据不断调整两个网络的参数，寻找最优策略（策略评估、策略提升）
    策略提升 target_net
    策略评估 eval_net
    """

    def __init__(self, state_dim, action_num, learning_rate, gamma, device, is_soft_update, update_fre=10,
                 update_ratio=1):
        self.state_dim = state_dim
        self.action_num = action_num
        self.target_net = Net(state_dim, action_num).to(device)
        self.eval_net = Net(state_dim, action_num).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.update_count = 0
        self.is_soft_update = is_soft_update
        self.update_fre = update_fre
        self.update_ratio = update_ratio
        self.device = device

    def choose_action(self, state, epsilon):
        """
        DQN 算法用 eval_net 网络按 e-greedy 生成行为策略，之后按此策略采样得到有关环境的消息 (s, a, r, s_)
        e-greedy 生成的行为策略，在 s 以 epsilon 的概率随机选择一个动作，以 1-epsilon 的概率选择 q(s, a) 最大的动作
        """
        if np.random.uniform() < epsilon:
            # 随机选择动作
            action = np.random.randint(self.action_num)
        else:
            # 选择 q(s, a) 最大的动作
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.eval_net(state)
            action = torch.argmax(q_values).item()
        return action

    def update(self, buffer: ReplayBuffer, batch_size):
        """
        DQN 算法利用经验回放区的数据按随机梯度下降的方式更新 eval_net（策略评估）
        更新一定次数后，认为 eval_net 已经很好的逼近新的目标策略（策略评估）
        此时，将其参数赋值给 target_net，让 target_net 作为新目标策略的值函数逼近，并生成新的目标策略（策略提升）
        之后用 eval_net 再评估新的目标策略，重复，直至收敛至最优
        """
        if buffer.size < batch_size:
            return

        batch_s, batch_a, batch_r, batch_s_, batch_done = buffer.sample(batch_size)
        # print(batch_s)
        # print(batch_a)
        # print(batch_r)
        # print(batch_s_)

        # 转变为 tensor 计算梯度
        batch_s = torch.tensor(batch_s, dtype=torch.float32).to(self.device)
        # 将 batch_a 变为二维，之后 gather 需要
        batch_a = torch.tensor(batch_a, dtype=torch.int64).to(self.device).unsqueeze(-1)
        # print(batch_a.shape)
        batch_r = torch.tensor(batch_r, dtype=torch.float32).to(self.device).unsqueeze(-1)
        batch_done = torch.tensor(batch_done, dtype=torch.int64).to(self.device).unsqueeze(-1)
        batch_s_ = torch.tensor(batch_s_, dtype=torch.float32).to(self.device)

        # eval_net 对 q(s, a) 的评估
        bat_ev_q_v = self.eval_net(batch_s).gather(1, batch_a)
        # print(bat_ev_q_v.shape)

        # q(s, a)真实值， 由目标策略给出
        bat_nex_tar_max_q_v = self.target_net(batch_s_).max(dim=1, keepdim=True)[0]
        bat_q_v = batch_r + (1 - batch_done) * self.gamma * bat_nex_tar_max_q_v
        # print(bat_q_v.shape)

        # 误差
        bat_loss = F.mse_loss(bat_ev_q_v, bat_q_v)

        # 梯度下降
        self.optimizer.zero_grad()
        bat_loss.backward()
        # 在反向传播算出梯度之后，参数更新之前，做这一步防止梯度爆炸或梯度消失
        nn.utils.clip_grad_value_(self.eval_net.parameters(), 100)
        self.optimizer.step()

        if not self.is_soft_update:
            self.update_count += 1
            if self.update_count % self.update_fre == 0:
                # eval_net 网络的值赋给 target_net（策略提升）
                self.target_net.load_state_dict(self.eval_net.state_dict())
        else:
            # 使用另一种方式更新 target_net
            # 即只评估一次就认为已经很好的近似，但为了更加正确的逼近，使用 eval_net 与旧策略的 target_net 加权平均
            # 加权平均的结果认为是更好的逼近
            eval_net_state_dict = self.eval_net.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            for key in eval_net_state_dict:
                target_net_state_dict[key] = eval_net_state_dict[key] * self.update_ratio + target_net_state_dict[
                    key] * (1 - self.update_ratio)
            self.target_net.load_state_dict(target_net_state_dict)

    def save_net(self, sav_dir='', sav_stru=False):
        if sav_stru:
            # 保存模型的结构和参数
            torch.save(self.eval_net, sav_dir + '/' + 'net.pkl')
        else:
            # 只保存模型的参数
            torch.save(self.eval_net.state_dict(), sav_dir + '/' + 'net_params.pkl')
