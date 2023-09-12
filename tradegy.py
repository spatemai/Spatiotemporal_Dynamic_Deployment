# 此文件用来画轨迹图
import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.xlim(0, 100)
plt.ylim(0, 100)

node_ini=np.load('result\\node_final2.npy')*10
user=np.load("data\\user_ini_final.npy",)*10

agent=np.load('result\\agent_final2.npy')*10

#plt.scatter(user[:,0], user[:,1])


img=plt.imread('C:\\Users\\98509\\Desktop\\研一\\研1下\\论文\\插图\\2D背景.png')
fig,ax=plt.subplots()
ax.imshow(img,extent=[0,1000,0,1000])

for i in range(50):
    l1,=plt.plot(user[0:10, i, 0], user[0:10, i, 1], color='y')
for i in range(10):
    l2,=plt.plot(node_ini[:, i, 0], node_ini[:, i, 1], color='blue')
l3,=plt.plot(agent[:,0,0], agent[:,0,1],color='red')
plt.plot(agent[:, 1, 0], agent[:, 1, 1],color='r')
plt.legend(handles=[l1,l2,l3],labels=['Ground perception node','Ground communication node','UAV communication node'],loc=4)
plt.show()