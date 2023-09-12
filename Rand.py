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

point=0
best=-100000
best_fugailv=0
best_shiyan=10000
best_episode=0
best_action=[]
best_data=[]
a_dim = env.a_dim
if __name__ == '__main__':


    MAX_EPISODES =10000
    MAX_EP_STEPS = 10
    rewardList = []
    fugailv = []
    shiyan = []

    for i in range(MAX_EPISODES):

        s = env.reset()
        ep_reward = 0
        tempt_action = []
        a = np.clip(np.random.rand(a_dim), -1, 1)  # 在动作选择上添加随机噪声
        for j in range(MAX_EP_STEPS):
            #if RENDER:
            #    env.render()

            # Add exploration noise
            #a = ddpg.choose_action(s.copy())

            #a = np.clip(np.random.rand(a_dim), -1, 1)  # 在动作选择上添加随机噪声

            #a=np.clip(a+np.random.random(a.size)*2-1, -1, 1)
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
                    best_fugailv = info[4]
                    best_shiyan = info[7]

                #if ep_reward > -300: RENDER = True
                break
            s = env.reset()

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
    np.save("result\\RANDreward_final.npy", rewardList)
    np.save("result\\RANDfugailv_final.npy", fugailv)
    np.save("result\\RANDshiyan_final.npy", shiyan)

