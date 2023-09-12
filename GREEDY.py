import numpy as np

import environment_rand as env
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
torch实现DDPG算法
"""
#import torch
#import numpy as np

import matplotlib.pyplot as plt
'''
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)
'''
MAX_EP_STEPS = 10
user=env.user[0:MAX_EP_STEPS]

point=0
best=-100000
best_episode=0
best_action=[]
best_data=[]
a_dim = env.a_dim

node=np.zeros((env.size,2))
agent=np.zeros((env.agent_n,3))




if __name__ == '__main__':


    MAX_EPISODES =10
    fugailv = []
    shiyan = []

    rewardList = []

    for i in range(MAX_EPISODES):

        s = env.reset()
        ep_reward = 0
        tempt_action = []
        #jj=np.sum(user[0,:,0])/50
        for j in range(MAX_EP_STEPS):
            #if RENDER:
            #    env.render()

            # Add exploration noise
            #a = ddpg.choose_action(s.copy())


            group = int(env.user_n / env.size)
            for k in range(env.size):
                node[k, 0] = np.sum(user[j,k * group:(k + 1) * group, 0]) / group
                node[k, 1] = np.sum(user[j,k * group:(k + 1) * group, 1]) / group

            group=int(env.size/env.agent_n)
            for k in range(env.agent_n):
                agent[k, 0] = np.sum(node[k * group:(k + 1) * group, 0]) / group
                agent[k, 1] = np.sum(node[k * group:(k + 1) * group, 1]) / group

            distance = np.zeros((env.agent_n, env.size))
            for ii in range(env.agent_n):
                for jj in range(env.size):
                    distance[ii, jj] = np.sqrt(
                        (agent[ii, 0] - node[jj, 0]) ** 2 + (agent[ii, 1] - node[jj, 1]) ** 2)
            for ii in range(env.agent_n):
                agent[ii,2] = np.max(distance[ii])

            agent[:,0:2]=agent[:,0:2]/50-1
            agent[:,2]=(agent[:,2]-env.z_min)/(env.z_max-env.z_min)*2-1
            node=node/50-1
            a=np.concatenate((agent.flatten(),node.flatten()))
            a = np.clip(a, -1, 1)


            #a=np.clip(a+np.random.random(a.size)*2-1, -1, 1)
            #a = np.clip(np.random.rand(a_dim), -1, 1)  # 在动作选择上添加随机噪声
            s_, r, done, info = env.step(a.copy())


            ep_reward += r
            tempt_action.append(info[3])
            if j == MAX_EP_STEPS - 1:
                #print('Episode:', i, ' Reward: ' , (ep_reward),  'r1',info[0], 'link',info[1], 'r3',info[2],'uav',info[3])
                if ep_reward>best:
                    best=ep_reward
                    #best_data=info[3]
                    best_action=tempt_action
                    best_episode=i
                    #best_fugailv = info[2]
                    best_fugailv = info[4]  #应该是这个
                    best_shiyan = info[7]
                #if ep_reward > -300: RENDER = True
                break

        rewardList.append(best)
        fugailv.append(best_fugailv)
        shiyan.append(best_shiyan)

    #print('Running time: ', time.time() - t1)
    print('best:',best_episode,best,best_action)
    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(fugailv)), fugailv)
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(shiyan)), shiyan)
    plt.show()
    rewardList = np.array(rewardList)

    np.save("result\\GREEDYreward.npy", rewardList)
    np.save("result\\GREEDYfugailv_final.npy", fugailv)
    np.save("result\\GREEDYshiyan_final.npy", shiyan)

