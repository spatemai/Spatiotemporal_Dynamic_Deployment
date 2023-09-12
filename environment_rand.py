import numpy as np

#T为时隙上限
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
#无人机最大移动距离
move_max=50
#电台最大移动距离
move_node=30
#传输数据相关数据
B=10000000#10Mhz
P_Tr=0.1#0.1W
g0=0.000142#1.42*10^(-4)
G0=2.2846#2.2846
sigma=0.000000000001#-90dbm=10^(-12)w
alpha=g0*G0/(sigma)

#电台的覆盖范围
coverage=30
#环境包括：每个用户的位置以及数据量、电台坐标、无人机坐标
s_dim=user_n*2+user_n*1+size*2+agent_n*3
#动作包括：无人机球坐标+电台球坐标
a_dim=agent_n*3+size*2

#(agent_n,3)
agent_ini=np.load('data\\agent_ini_final.npy')
agent_ini_nor=np.zeros_like(agent_ini)
agent_ini_nor[:,0:2]=agent_ini[:,0:2]/x*2-1
agent_ini_nor[:,2]=(agent_ini[:,2]-z_min)/(z_max-z_min)*2-1
#(size,2)
node_ini=np.load('data\\node_ini_final.npy')
node_ini_nor=np.zeros_like(node_ini)
node_ini_nor=node_ini/x*2-1
#(T,user_n,1)
user_data=np.load('data\\user_data_ini_final.npy')
#(T,user_n,2)
user=np.load('data\\user_ini_final.npy')

agent=np.zeros_like(agent_ini)
node=np.zeros_like(node_ini)


#状态s包括：节点坐标、节点数据大小    ##无人机坐标、
def reset():
    global agent
    global node

    agent=agent_ini.copy()
    node=node_ini.copy()

    '''
    
    agent = np.random.random((agent_n, 3))
    agent[:, 0] = agent[:, 0] * x
    agent[:, 1] = agent[:, 1] * y
    agent[:, 2] = agent[:, 2] * z
    '''
    #尝试将环境标准化试试(-1,1)
    #s = np.concatenate((node.flatten(), node_data.flatten() ))
    global t
    t=0

    s = np.concatenate((user[0].flatten()/50-1, (user_data[0].flatten()-100000*8)/(900000*8)*2-1,agent_ini_nor.flatten(),node_ini_nor.flatten()))

    #s:(size*3+size*1)
    return s.copy()

#动作包括：无人机坐标、每个电台与哪个无人机相连接、(agent_n*3)
#奖励包括：覆盖率以及延迟（直接保证覆盖率为100%）
#输入的数据是经过标准化的(-1,1)
def step(a):
    '''
    #提取节点坐标，注意节点经过标准化,(size,3)
    node_location_=s[:size*3].reshape((size,3))*50+50
    node_location_[:,2]=0
    #提取节点数据大小，(size,1)
    node_data_=((s[size*3:].reshape((size,1))+1)/2)*900000*8+100000*8
    '''


    #提取无人机坐标(agent_n,3)
    #agent_location_ = a[:agent_n * 3].copy()
    #agent_location_ = a.copy()
    # 先获取当前的时间t
    global t
    global agent
    global node
    user_cov=user_n

    user_linked=np.zeros((user_n,size))
    renwu=np.zeros(size)
    # 提取无人机移动向量(agent_n,3),动作为球面坐标系(r,theta,phi)
    # agent_location_ = a[:agent_n * 3].copy()
    agent_move = a[:agent_n*3].copy().reshape((agent_n, 3))
    # 真实的移动距离
    move_real = (agent_move[:, 0] + 1) / 2 * move_max
    # 利用球形坐标系来计算无人机的移动后坐标位置  垂直角度为：0-pi
    agent_x = move_real * np.sin(np.pi * (agent_move[:, 2] + 1) / 2) * np.cos(np.pi * agent_move[:, 1])
    agent_y = move_real * np.sin(np.pi * (agent_move[:, 2] + 1) / 2) * np.sin(np.pi * agent_move[:, 1])
    agent_z = move_real * np.cos(np.pi * (agent_move[:, 2] + 1) / 2)
    agent[:, 0] = agent[:, 0] + agent_x
    agent[:, 1] = agent[:, 1] + agent_y
    agent[:, 2] = agent[:, 2] + agent_z

    # 无人机的坐标不应该超出边界，因此在这里修正一下
    agent[agent < 0] = 0
    agent[agent > x] = x
    agent[:, 2][agent[:, 2] < z_min] = z_min
    agent[:, 2][agent[:, 2] > z_max] = z_max





    #提取电台的坐标(size,2)
    node_move=a[agent_n * 3:agent_n * 3+size*2].copy()
    node_move=node_move.reshape((size,2))
    node[:,0]=node[:,0]+move_node*(node_move[:,0]+1)/2*np.cos(np.pi*node_move[:,1])
    node[:, 1] = node[:, 1] + move_node * (node_move[:, 0] + 1) / 2 * np.sin(np.pi * node_move[:, 1])
    # 电台的坐标不应该超出边界，因此在这里修正一下
    node[node < 0] = 0
    node[node > x] = x

    # 提取所有用户与电台连接状态(user_n,size)
    link_1 = np.zeros((user_n, size))
    #link_1 = link_1.astype(np.int32)
    #计算所有电台和用户的距离(user_n,size)
    distance_1 = np.zeros((user_n, size))
    #d=distance_1.copy()
    #np.repeat()
    '''
    for i in range(user_n):
        for j in range(size):
            d[i, j] = np.sqrt(
                (user[t, i, 0] - node[j, 0]) ** 2 + (user[t, i, 1] - node[j, 1]) ** 2 )
    '''
    #
    for i in range(size):
        distance_1[:, i] = np.sqrt((user[t,:, 0] - node[i, 0]) ** 2 + (user[t,:, 1] - node[i, 1]) ** 2)


    #定义电台的数据量(size)
    node_data=np.zeros(size)
    # 对电台的所有连接状态进行检查，不允许超出覆盖范围仍然连接的情况发生
    #计算电台接收到的数据总量，这里采取的策略是和最近的用户相连接，如果超过覆盖范围的话就采取增大数据量的方式进行惩罚
    for i in range(user_n):
        #y=distance_1[i]
        #x=np.argmin(distance_1[i])
        #z=y[x]
        #np.take_along_axis[]
        node_data[np.argmin(distance_1[i])]=node_data[np.argmin(distance_1[i])]+user_data[t,i]
        renwu[np.argmin(distance_1[i])]=renwu[np.argmin(distance_1[i])]+user_data[t,i]
        user_linked[i, np.argmin(distance_1[i])] = 1
        if np.min(distance_1[i])>coverage:
            node_data[np.argmin(distance_1[i])]=node_data[np.argmin(distance_1[i])]+50000000
            user_cov=user_cov-1
            user_linked[i,np.argmin(distance_1[i])]=0
            renwu[np.argmin(distance_1[i])] = renwu[np.argmin(distance_1[i])] - user_data[t, i]




    #提取所有节点与无人机连接状态(size,1)
    link=np.zeros((size,1))
    link = link.astype(np.int32)
    '''
    link=a[agent_n*3:].copy()
    link=link.reshape((size, 1))
    #对连接状态进行修正，注意此处是两个无人机
    link[link<=0]=0
    link[link>0]=1
    link=link.astype(np.int32)
    '''
    #计算所有节点与无人机的距离(size,agent_n)
    distance=np.zeros((size,agent_n))
    for i in range(size):
        for j in range(agent_n):
            distance[i,j]=np.sqrt((node[i,0]-agent[j,0])**2+(node[i,1]-agent[j,1])**2+(agent[j,2])**2 )
    #对所有连接状态进行检查，不允许超出覆盖范围仍然连接的情况发生


    #此处令覆盖范围等于高度(45°)
    for i in range(size):
        if (distance[i,0]>agent[0,2]*np.sqrt(2) and distance[i,1]>agent[1,2]*np.sqrt(2)):
            link[i]=-1
            user_linked[:, i] = 0
            renwu[i]=0
            continue
        if (distance[i,0]<=agent[0,2]*np.sqrt(2) and distance[i,1]>agent[1,2]*np.sqrt(2)):
            link[i]=0
            continue
        if (distance[i,0]>agent[0,2]*np.sqrt(2) and distance[i,1]<=agent[1,2]*np.sqrt(2)):
            link[i]=1
            continue
        if (distance[i,0]<distance[i,1]):
            link[i]=0
        else:
            link[i]=1



    #传输速率(size,1) 为了凸显距离的重要性，将距离*10
    rate=np.zeros((size,1))
    for i in range(size):
        if link[i]>-1:#连接状态的节点传输速率
            rate[i]=B*np.log2(1+alpha*P_Tr/(distance[i,link[i]]**2 *1000))
        else:#否则直接赋值为最大传输距离(变相令不连接的节点延迟高)，
            rate[i] = B * np.log2(1 + alpha * P_Tr / (x**2+y**2+z**2) / 1000)/10


    r1=np.sum(node_data / rate)#总延迟，奖励
    #r2=np.amax(node_data[t] / rate) * (np.sum(link < 0))/3
    r3=np.sum(link >= 0 )/size#电台覆盖率
    #r4=user_cov/user_n#用户覆盖率
    r4 = np.sum(user_linked) / user_n  # 用户覆盖率
    r5=np.sum(renwu / rate)#真正延迟


    reward=-np.sum(node_data/rate)/50
    #reward =(np.sum(link >= 0)) /size

    #动作标准化
    agent_nor = np.zeros_like(agent)
    agent_nor[:, 0:2] = agent[:, 0:2] / x * 2 - 1
    agent_nor[:, 2] = (agent[:, 2] - z_min) / (z_max - z_min) * 2 - 1
    node_nor = np.zeros_like(node)
    node_nor = node / x * 2 - 1


    s_ = np.concatenate((user[t].flatten() / 50 - 1, (user_data[t].flatten() - 100000 * 8) / (900000 * 8) * 2 - 1,agent_nor.flatten(),node_nor.flatten()))

    t = 0
    #s_, r, done, info
    return [s_.copy(),reward,[],[r1,link,r3,agent.copy(),r4,node_data,node.copy(),r5]]

