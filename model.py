import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
T = 50
t=0
#随机化电台数量
size = 10
#无人机数量
agent_n=2
#随机化用户数量
user_n = 50
#三维坐标范围 高度更改为20-100
x=100
y=100
np.random.seed(1)
# Fixing random state for reproducibility
# np.random.seed(19680801)
node = np.random.random((T+1,size,3))
node[:,:,2]=node[:,:,2]*0

#用户的传输数据大小[100KB,1000KB]  (T,user_n,1)
user_data=np.random.random((T+1,user_n,1))*900000*8+100000*8
user_data=np.load('user_data_ini.npy')
#初始化用户位置(T,user_n,3)
ue = np.random.random((T+1,user_n,3))
ue[:,:,0:2]=np.load('user_ini.npy')/100
ue[:,:,2]=ue[:,:,2]*0


def random_walk(num_steps, max_step=0.05, stype=0):  # walk次数，walk最大步长,stype标注生成的是粒子（0）还是agent（1）,默认是粒子
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)  # 随机坐标

    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))  # 正态分布抽取num_step 次行走的次数
    walk = np.zeros_like(steps)
    walk[0] = start_pos
    # walk = start_pos + np.cumsum(steps, axis=0)#cumsum是累加求和
    for i in range(num_steps - 1):
        walk[i + 1] = walk[i] + steps[i]
        walk[walk < 0] = 0
        walk[walk > 1] = 1

    if stype == 0:

        #walk[:,0:2]=np.array([[4.17022005e+01,7.20324493e+01], [1.14374817e-02, 3.02332573e+01], [1.46755891e+01 ,9.23385948e+00], [1.86260211e+01, 3.45560727e+01], [3.96767474e+01, 5.38816734e+01], [4.19194514e+01, 6.85219500e+01], [2.04452250e+01, 8.78117436e+01], [2.73875932e+00, 6.70467510e+01], [4.17304802e+01 ,5.58689828e+01], [1.40386939e+01, 1.98101489e+01], [8.00744569e+01, 9.68261576e+01], [3.13424178e+01, 6.92322616e+01], [8.76389152e+01, 8.94606664e+01], [8.50442114e+00, 3.90547832e+00], [1.69830420e+01, 8.78142503e+01], [9.83468338e+00, 4.21107625e+01], [9.57889530e+01, 5.33165285e+01], [6.91877114e+01 ,3.15515631e+01], [6.86500928e+01 ,8.34625672e+01], [1.82882773e+00 ,7.50144315e+01]])
        walk[:, 2] = 0
    else:

        walk[:, 2] = 0.8
    return walk  # walk次数个三维坐标


def coverage(x, y, z, r):  # 圆锥圆心的坐标(x,y,z)以及映射到地面的半径r

    u = np.linspace(0, 2 * np.pi, 50)  # 生成等差数列
    v = np.linspace(0, np.pi, 50)
    X = np.outer(np.cos(u), np.sin(v))  # outer(a,b)外积：a的每个元素乘以b的每个元素，二维数组
    Y = np.outer(np.sin(u), np.sin(v))
    Z = 1 - np.sqrt(X ** 2 + Y ** 2)

    X = X * r + x
    Y = Y * r + y
    Z = Z * z
    return X, Y, Z


def update_lines(num, walks, lines):
    print(num)

    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        # .set_data()意思是更新其中数据
        line.set_data(walk[num, :2].T)
        line.set_3d_properties(walk[num, 2])
    '''
    for i in agent:
        x,y,z=coverage(i[num,0],i[num,1],i[num,2],0.3)
        ax.plot_surface(x,y,z,alpha=0.5)

    cov[0].remove()
    cov[1].remove()
    cov[2].remove()
    x, y, z = coverage(agent[0][num, 0], agent[0][num, 1], agent[0][num, 2], 0.3)
    cov[0]=ax.plot_surface(x, y, z, alpha=0.5)
    '''
    for i in range(agent_n):
        cov[i].remove()

    for i in range(agent_n):
        x, y, z = coverage(agent[i][num, 0], agent[i][num, 1], agent[i][num, 2], agent[i][num, 2])
        cov[i] = ax.plot_surface(x, y, z, alpha=0.1, color='red')

    for i in range(size):
        cov_node[i].remove()

    for i in range(size):
        x, y, z = coverage(user_np[i][num, 0], user_np[i][num, 1], 0, 0.3)
        cov_node[i] = ax.plot_surface(x, y, z, alpha=0.1, color='blue')







    return lines


# Data: 40 random walks as (num_steps, 3) arrays
num_steps = 10
#size = 10  # 粒子个数
user_n = 50
agent_n = 2  # agent个数
agent = [random_walk(num_steps, stype=1) for index in range(agent_n)]
user = [random_walk(num_steps, ) for index in range(user_n)]
walks = [random_walk(num_steps) for index in range(size)]  # list:粒子个数*array(walk次数*三维坐标)

agent_=np.load('agent.npy')/100

agent=[agent_[:,0],agent_[:,1]]

user_np=np.zeros((size,10,3))
user_=np.load('node.npy')

for i in range(size):
    user_np[i,:,0:2]=user_[:,i,:]
user_np=user_np/100
user=[user_np[index] for index in range(size)]
walks=[ue[0:10,index] for index in range(user_n)]

walks.extend(user)
walks.extend(agent)

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# 创建圆锥
# x=agent[0][:,0]
# X,Y,Z=coverage(0.5,0.5,0.8,0.2)


# Create lines initially without data
lines = [ax.plot([], [], [], 'o', color='y')[0] for index in range(user_n)]
lines = lines + [ax.plot([], [], [], 'o', color='b')[0] for index in range(size)]
lines = lines + [ax.plot([], [], [], 'o', color='r')[0] for index in range(agent_n)]

tempt1, tempt2, tempt3 = coverage(agent[0][0, 0], agent[0][0, 1], agent[0][0, 2], agent[0][0, 2])
cov = [ax.plot_surface(tempt1, tempt2, tempt3, alpha=0.1, color='r') for index in range(agent_n)]

tempt3, tempt4, tempt5 = coverage(user_np[0][0, 0], user_np[0][0, 1], 0, 0.1)
cov_node = [ax.plot_surface(tempt3, tempt4, tempt5, alpha=0.1, color='b') for index in range(size)]


# Setting the axes properties
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=2000,cache_frame_data=False)

plt.show()