import numpy as np

import environment as env
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
best_episode=0
best_action=[]
best_data=[]

if __name__ == '__main__':


    MAX_EPISODES =10
    MAX_EP_STEPS = 1
    rewardList = []

    for i in range(MAX_EPISODES):

        s = env.reset()
        ep_reward = 0
        tempt_action = []
        for j in range(MAX_EP_STEPS):
            #if RENDER:
            #    env.render()

            # Add exploration noise
            #a = ddpg.choose_action(s.copy())

            a = np.array([57.8749,61.32162,59.13717,29.52431729,20.36452226,33.42543121])/50-1  # 在动作选择上添加随机噪声
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
                #if ep_reward > -300: RENDER = True
                break

        rewardList.append(best)

    #print('Running time: ', time.time() - t1)
    print('best:',best_episode,best,best_action)
    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()

