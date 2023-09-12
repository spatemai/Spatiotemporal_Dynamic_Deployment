#创建初始数据，包括无人机初始坐标、电台初始坐标、用户每个时隙的数据量、用户每个时隙的坐标
#解决方案：先进行三维部署，随后每个时隙进行轨迹布置
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(111)   # 随机种子代号自己设定
T =50
t=0
#随机化电台数量
size=10
#随机化用户数量
user_n = 50
#无人机数量
agent_n=2
#三维坐标范围 高度更改为20-80
x=100
y=100
z=80
z_max=80
z_min=20
#初始化电台位置(size,2)
#之后可以计算出最佳位置后来代替，此时暂时使用随机位置
node_ini = np.random.random((size,2))
node_ini[:,0]=node_ini[:,0]*x
node_ini[:,1]=node_ini[:,1]*y

#node记录电台的实时位置
node=node_ini

#用户的传输数据大小[100KB,1000KB]  (T,user_n,1)
user_data=np.random.random((T+1,user_n,1))
#初始化用户位置(T,user_n,3)
user_step=10#用户每个时隙移动距离为10
user_walk=np.random.random((T+1,user_n,2))#按照极坐标定义每步
user = np.zeros((T+1,user_n,2))#定义用户的位置信息
#定义用户第一个时隙的位置
#随后使用随机移动
user[0]=np.random.random((user_n,2))
user[0,:,0]=user[0,:,0]*x
user[0,:,1]=user[0,:,1]*y
for i in range(T):
    tempt_x=user[i,:,0].copy()
    tempt_y=user[i,:,1].copy()
    user[i+1,:,0]=user[i,:,0]+user_step*user_walk[i,:,0]*np.cos(user_walk[i,:,1]*np.pi*2)
    user[i+1,:,1]=user[i,:,1]+user_step*user_walk[i,:,0]*np.sin(user_walk[i,:,1]*np.pi*2)
    user[i + 1, :, 0][user[i+1,:,0]<0]=tempt_x[user[i+1,:,0]<0]
    user[i + 1, :, 0][user[i + 1, :, 0] >x] = tempt_x[user[i + 1, :, 0] >x]
    user[i + 1, :, 1][user[i + 1, :, 1] < 0] = tempt_y[user[i + 1, :, 1] < 0]
    user[i + 1, :, 1][user[i + 1, :, 1] > x] = tempt_x[user[i + 1, :, 1] > x]
    #user[user<0]=-user[user<0]
    #user[user>x]=2*x-user[user>x]

#电台的覆盖范围
coverage=30

#初始化无人机位置(agent_n,3)
#此处使用随机位置，之后可以通过算法计算最佳的初始位置
agent=np.random.random((agent_n,3))
agent[:,0]=agent[:,0]*x
agent[:,1]=agent[:,1]*y
agent[:, 2] = agent[:, 2] * (z_max-z_min) + z_min


s_dim=user_n*3+user_n*1
a_dim=agent_n*3+size*2

#作图，画出初始情况
#画图

plt.figure()
plt.xlim(0,100)
plt.ylim(0, 100)
for i in range(user_n):
    plt.scatter(user[0,i,0], user[0,i,1], c='y', alpha=user_data[0,i]/2+0.5)#
#plt.scatter(node[0,:,0],node[0,:,1], alpha=node_data[0,i])
banjing=agent[:,2]

for i in range(agent_n):
    plt.scatter(agent[i,0], agent[i,1], color='r', alpha=1)
    plt.scatter(np.linspace(agent[i,0] - banjing[i], agent[i,0] + banjing[i], 10000),
                agent[i,1] - np.sqrt(banjing[i] ** 2 - (
                        np.linspace(agent[i,0] - banjing[i], agent[i,0] + banjing[i], 10000) - agent[i,0]) ** 2),color='r',s=0.001)
    plt.scatter(np.linspace(agent[i, 0] - banjing[i], agent[i, 0] + banjing[i], 10000),
                agent[i, 1] + np.sqrt(banjing[i] ** 2 - (
                        np.linspace(agent[i, 0] - banjing[i], agent[i, 0] + banjing[i], 10000) - agent[i, 0]) ** 2),
                color='r', s=0.001)

for i in range(size):
    plt.scatter(node[i,0], node[i,1], color='b', alpha=1)
    plt.scatter(np.linspace(node[i,0] - coverage, node[i,0] + coverage, 10000),
                node[i,1] - np.sqrt(coverage ** 2 - (
                        np.linspace(node[i,0] - coverage, node[i,0] + coverage, 10000) - node[i,0]) ** 2),color='b',s=0.001)
    plt.scatter(np.linspace(node[i, 0] - coverage, node[i, 0] + coverage, 10000),
                node[i, 1] + np.sqrt(coverage ** 2 - (
                        np.linspace(node[i, 0] - coverage, node[i, 0] + coverage, 10000) - node[i, 0]) ** 2),
                color='b', s=0.001)
plt.show()
user_data=user_data*900000*8+100000*8
np.save("data\\agent_ini_final.npy",agent)
np.save("data\\node_ini_final.npy",node_ini)
np.save("data\\user_ini_final.npy",user)
np.save("data\\user_data_ini_final.npy",user_data)
