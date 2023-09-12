import numpy as np
import environment as env
#import environment_rand as env
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
torch实现DDPG算法
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
'''
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)
'''
# 定义训练的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

point=0
best=-100000
best_episode=0
best_action=[]
best_data=[]
# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.FloatTensor(action_bound).to(device)

        # layer
        self.layer_1 = nn.Linear(state_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        # nn.init.normal_(self.layer_1.weight, 0., 0.3)
        # nn.init.constant_(self.layer_1.bias, 0.1)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1)
        self.layer_2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        # nn.init.normal_(self.layer_2.weight, 0., 0.3)
        # nn.init.constant_(self.layer_2.bias, 0.1)

        self.layer_3 = nn.Linear(512, 256)
        # nn.init.normal_(self.layer_3.weight, 0., 0.3)
        # nn.init.constant_(self.layer_3.bias, 0.1)
        self.ln3 = nn.LayerNorm(256)

        self.layer_4 = nn.Linear(256, 128)
        # nn.init.normal_(self.layer_3.weight, 0., 0.3)
        # nn.init.constant_(self.layer_3.bias, 0.1)
        self.ln4 = nn.LayerNorm(128)

        self.output = nn.Linear(128, action_dim)
        # self.output.weight.data.normal_(0., 0.3)
        # self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = F.relu(self.ln1(self.layer_1(s)))
        a = F.relu(self.ln2(self.layer_2(a)))
        a = F.relu(self.ln3(self.layer_3(a)))
        a = F.relu(self.ln4(self.layer_4(a)))
        a = torch.tanh(self.output(a))
        # 对action进行放缩，实际上a in [-1,1]
        # a = self.bn(a)
        # scaled_a = a * self.action_bound
        # scaled_a = a
        # return scaled_a
        return a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 1024
        # layer
        self.layer_1 = nn.Linear(state_dim + action_dim, n_layer)
        # nn.init.normal_(self.layer_1.weight, 0., 0.1)
        # nn.init.constant_(self.layer_1.bias, 0.1)
        self.ln1 = nn.LayerNorm(1024)

        self.layer_2 = nn.Linear(1024, 512)
        # nn.init.normal_(self.layer_2.weight, 0., 0.1)
        # nn.init.constant_(self.layer_2.bias, 0.1)
        self.ln2 = nn.LayerNorm(512)

        self.layer_3 = nn.Linear(512, 258)
        # nn.init.normal_(self.layer_3.weight, 0., 0.1)
        # nn.init.constant_(self.layer_3.bias, 0.1)
        self.ln3 = nn.LayerNorm(258)

        self.layer_4 = nn.Linear(258, 128)
        # nn.init.normal_(self.layer_4.weight, 0., 0.1)
        # nn.init.constant_(self.layer_4.bias, 0.1)
        self.ln4 = nn.LayerNorm(128)

        self.output = nn.Linear(128, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.layer_1(x)
        x = self.layer_2(F.relu(self.ln1(x)))
        x = self.layer_3(F.relu(self.ln2(x)))
        x = self.layer_4(F.relu(self.ln3(x)))
        # x = self.layer_4(F.relu(x))
        # a = self.layer_2(a)
        q_val = self.output(F.relu(self.ln4(x)))
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacity=1000, gamma=0.9, lr_a=0.000001,
                 lr_c=0.0001, batch_size=128):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss().to(device)

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(device)
        action = self.actor(s)
        return action.detach().cpu().numpy()

    def learn(self):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0] + '.weight']
                al[1].weight.data.mul_((1 - tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
                al[1].bias.data.mul_((1 - tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1 - tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
                cl[1].bias.data.mul_((1 - tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])

        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

            self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:]).to(device)

        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        #a_loss.backward(retain_graph=True)
        a_loss.backward()
        self.aopt.step()

        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

        if point%100==0:
            print(-a_loss,td_error)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition.copy()
        self.pointer += 1



import time

if __name__ == '__main__':

    # hyper parameters
    VAR = 3 # control exploration
    MAX_EPISODES =8000
    MAX_EP_STEPS = 10
    MEMORY_CAPACITY = 10000
    REPLACEMENT = [
        dict(name='soft', tau=0.0008),
        dict(name='hard', rep_iter=100)
    ][0]  # you can try different target replacement strategies

    #ENV_NAME = 'Pendulum-v0'
    #RENDER = False

    rewardList=[]
    fugailv=[]
    shiyan=[]

    # train
    #env = gym.make(ENV_NAME)
    #env = env.unwrapped
    #env.seed(1)

    s_dim = env.s_dim
    a_dim = env.a_dim
    a_bound = 1
    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                action_bound=a_bound,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY,
                gamma=0.00001)

    t1 = time.time()
    for i in range(MAX_EPISODES):

        s = env.reset()
        ep_reward = 0
        tempt_action = []
        tempt_data = []
        tempt_node = []
        for j in range(MAX_EP_STEPS):
            #if RENDER:
            #    env.render()

            # Add exploration noise
            a = ddpg.choose_action(s.copy())

            a = np.clip(a+np.clip(np.random.normal(0, VAR,a.shape),-1,1), -1, 1)  # 在动作选择上添加随机噪声
            #a=np.clip(a+np.random.random(a.size)*2-1, -1, 1)
            s_, r, done, info = env.step(a.copy())
            xx=[s,s_]
            ddpg.store_transition(s, a, r , s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .9998  # decay the action randomness
                if VAR<=0.000001:
                    VAR=0.000001
                point = point + 1

                ddpg.learn()

            s = s_.copy()
            #s = env.reset()


            ep_reward += r
            tempt_action.append(info[3].copy())
            tempt_node.append(info[6])
            tempt_data.append(a.copy())
            if j == MAX_EP_STEPS - 1  :
                if i%100==0:
                    print('Episode:', i, ' Reward: ' , (ep_reward), 'Explore: ' , VAR, 'r1',info[0], 'link',info[1], '电台覆盖率',info[2],'用户覆盖率',info[4],'uav动作',a)
                if ep_reward>best:
                    best=ep_reward
                    best_data=tempt_data
                    best_node = tempt_node
                    best_action=tempt_action
                    best_episode=i
                #if ep_reward > -300: RENDER = True
                break

        rewardList.append(ep_reward)
        fugailv.append(info[4])
        shiyan.append(info[7])

    print('Running time: ', time.time() - t1)
    print('best:',best_episode,best,best_action,best_data)
    '''
    plt.figure()
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.scatter(done[:,0], done[:,1])
    final_action=np.array(best_action)
    plt.plot(final_action[:,0,0], final_action[:,0,1],color='red')
    plt.plot(final_action[:, 1, 0], final_action[:, 1, 1],color='green')
    plt.show()
    '''


    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(fugailv)), fugailv)
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(shiyan)), shiyan)
    plt.show()

    np.save("result\\node_final2.npy", best_node)
    np.save("result\\agent_final2.npy", best_action)
    np.save("result\\RLreward_ini_final2.npy", rewardList)
    np.save("result\\fugailv_final2.npy", fugailv)
    np.save("result\\shiyan_final2.npy", shiyan)

 # 保存模型参数至文件
    #torch.save(ddpg.actor.state_dict(), 'net_actor_parameters.pt')
    #torch.save(ddpg.critic.state_dict(), 'net_critic_parameters.pt')