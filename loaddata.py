import numpy as np
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体（解决中文无法显示的问题）
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号“-”显示方块的问题


ddpgreward1=np.load('result\\RLreward_ini_final1.npy')[:8000]
plt.figure()
l1,=plt.plot(np.arange(len(ddpgreward1)), ddpgreward1)

plt.xlabel("Training Epoch")
plt.ylabel("Score")
plt.legend(handles=[l1],labels=['DDPG'],loc='best')


plt.show()

randreward1=np.load('result\\RANDfugailv_final.npy')[:8000]
greedyreward1=np.load('result\\GREEDYfugailv_final.npy')
#greedyreward=np.tile(greedyreward[0],len(randreward))
ddpgreward1=np.load('result\\fugailv_final.npy')[:8000]
plt.figure()
l1,=plt.plot(np.arange(len(ddpgreward1)), ddpgreward1)
l2,=plt.plot(np.arange(len(randreward1)), randreward1)
l3,=plt.plot(np.arange(len(greedyreward1)*1000), np.repeat(greedyreward1,1000))
plt.xlabel("Training Epoch")
plt.ylabel("Coverage")
plt.legend(handles=[l1,l2,l3],labels=['DDPG','Rand','Cluster'],loc='best',prop={'size':15})


plt.show()
'''
randreward=np.load('result\\RANDfugailv_final.npy')
greedyreward=np.load('result\\GREEDYfugailv_final.npy')
#greedyreward=np.tile(greedyreward[0],len(randreward))
ddpgreward=np.load('fugailv_final.npy')
plt.figure()
plt.plot(np.arange(len(randreward)), randreward)
plt.plot(np.arange(len(greedyreward)), greedyreward)
plt.plot(np.arange(len(ddpgreward)), ddpgreward)


plt.show()
'''
ddpgreward1[ddpgreward1<=0]=1
randreward=np.load('result\\RANDshiyan_final.npy')[:8000]/randreward1/50
greedyreward=np.load('result\\GREEDYshiyan_final.npy')/greedyreward1/50
#greedyreward=np.tile(greedyreward[0],len(randreward))
ddpgreward=np.load('result\\shiyan_final.npy')[:8000]/ddpgreward1/50
plt.figure()
l1,=plt.plot(np.arange(len(ddpgreward)), ddpgreward)
l2,=plt.plot(np.arange(len(randreward)), randreward)
l3,=plt.plot(np.arange(len(greedyreward)*1000), np.repeat(greedyreward,1000))
plt.legend(handles=[l1,l2,l3],labels=['DDPG','Rand','Cluster'],loc='best')
plt.xlabel("Training Epoch")
plt.ylabel("Delay")

plt.show()