import numpy as np
import environment as env
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import sys
#np.random.seed(111)
size=5000#粒子的个数,,注意如果粒子个数太大的话，可能会一开始就随机到边界的解，这个时候由于惩罚因子很小，适应度会误认为是最优解。因此注意迭代次数如果太小需要警惕；粒子群数目小的时候注意速度上限可以适当设置大一点。。
N=2
#a=10#fitness的参数

#np.random.seed(0)

def q(x):
    y=x.copy()
    #相对惩罚函数
    y[x<0]=0
    return y

def theta(x):
    #分段赋值函数
    y=np.zeros_like(x)+300
    y[x<=1]=100
    y[x <= 0.1] = 20
    y[x < 0.001] = 10
    return y

def gamma(x):
    #惩罚指数
    y=np.zeros_like(x)+2
    y[x<1]=1
    return y

def H(x):
    #约束惩罚项
    tempt=theta(q(x))*np.power(q(x),gamma(q(x)))*100
    tempt=tempt.reshape(size,-1)
    return tempt

def fitness_value(inf,k):
    #return：自适应值
    value,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15=env.fitness(inf)
    #加入惩罚项
    h=k*math.sqrt(k)#动态变化的惩罚系数
    #限制条件：
    '''H1=H(x+y-6)
    H2=H(3*x-2*y-5)
    H3 = H(1-x)
    H4 = H(x-2)
    H5 = H(-1-y)
    H6 = H(y)
    Z=Z+h*(H1+H2+H3+H4+H5+H6)'''
    H1 = H(h1)
    H2 = H(h2)
    H3 = H(h3)
    H4 = H(h4)
    H5 = H(h5)
    H6 = H(h6)
    H7 = H(h7)
    H8 = H(h8)
    H9 = H(h9)
    H10 = H(h10)
    H11 = H(h11)
    H12 = H(h12)
    H13= H(h13)
    H14 = H(h14)
    H15 = H(h15)


    value = value + 0 * (np.sum(H1,axis=1)+np.sum(H2,axis=1)+np.sum(H3,axis=1)+np.sum(H4,axis=1)+np.sum(H5,axis=1)+np.sum(H6,axis=1)+np.sum(H7,axis=1)+np.sum(H8,axis=1)+np.sum(H9,axis=1)+np.sum(H10,axis=1)+np.sum(H11,axis=1)+np.sum(H12,axis=1)+np.sum(H13,axis=1)+np.sum(H14,axis=1)+np.sum(H15,axis=1))

    return value

def velocity_update(w,V,c1,c2,pbest,gbest,X,V_max):
    #return:更新速度
    size = X.shape[0]  # 返回矩阵X的行数
    r1 = np.random.random((size, 1))  # 该函数表示成size行 1列的浮点数，浮点数都是从0-1中随机。
    r2 = np.random.random((size, 1))
    V =w*V+c1*r1*(pbest-X)+c2*r2*(gbest-X)
    for i in range(6*N):
        V[:,i][V[:,i] > V_max[i]] = V_max[i]  # 防止速度超过上限
        V[:,i][V[:,i] < -V_max[i]] = -V_max[i]

    return V


def pos():
    Time=10000#迭代轮次

    w_ini=1.6
    w_end=0.4

    #s_dim = env.s_dim
    a_dim = 3




    # 初始化各个粒子
    s=env.reset()

    X=np.random.random((size,a_dim*N))*2-1
    # 初始化各个粒子的位置
    #X = np.random.uniform(0, 100, size=(size, dim))
    # 初始化各个粒子的速度
    V = np.zeros_like(X)
    fitness_best_list = []  # 记录粒子群适应度最优值


    c1=2#学习因子
    c2=2#学习因子
    #loc_N,P_n,B_n,r_n 100000,100000,500,100,10000000,100000
    V_max=0.2*np.ones_like(X) #速度上限,0.1/1*范围区间，可以每一维度都设置独特的速度上限
    #print(X)
    chi=0.8#限制速度的限制因子 0.76

    #inf=[loc_N, P_n, B_n, r_n]
    p_fitness = env.step(inf,1)  #每个粒子的最优适应度

    g_fitness = p_fitness.min()  #粒子群的最优适应度
    fitness_best_list.append(g_fitness)

    pbest=X.copy()#粒子发现最好位置
    gbest=pbest[p_fitness.argmin()].copy()#粒子群发现最好位置


    for i in range(Time):
        w = (w_ini - w_end) * (Time - i) / Time + w_end#动态调整w：惯性因子
        #w=1
        V=chi*velocity_update(w,V,c1,c2,pbest,gbest,X,V_max)#更新速度
        X=X+V#更新粒子的位置

        loc_N=X[:,:N*3].copy()
        P_n=X[:,N*3:N*3+N].copy()
        B_n=X[:,N*3+N:N*3+N+N].copy()
        r_n=X[:,N*3+N+N:N*3+N+N+N].copy()
        inf=[loc_N.reshape(shape1),P_n.reshape(shape2),B_n.reshape(shape3),r_n.reshape(shape4)]

        fitness = fitness_value(inf,i+1)  # 更新每个粒子的适应度

        tempt=p_fitness>fitness
        #print(tempt)
        pbest[tempt]=X[tempt].copy() #更新每个粒子发现的最好位置
        p_fitness[tempt] = fitness[tempt].copy()#更新每个粒子的最优适应度
        gbest = pbest[p_fitness.argmin()].copy()  #更新粒子群发现最好位置

        #test
        X_best = gbest[np.newaxis, :].copy()
        loc_N_best = X_best[:, :3*N].copy()
        P_n_best = X_best[:, 3*N:4*N].copy()
        B_n_best = X_best[:, 4*N:5*N].copy()
        r_n_best = X_best[:, 5*N:6*N].copy()
        shape1_best = (1,N,3)
        shape2_best = (1,N)
        shape3_best = (1,N)
        shape4_best = (1,N)
        information = [loc_N_best.reshape(shape1_best), P_n_best.reshape(shape2_best), B_n_best.reshape(shape3_best), r_n_best.reshape(shape4_best)]
        value=env.getinf(information)

        g_fitness = p_fitness.min()  # 粒子群的最优适应度
        fitness_best_list.append(g_fitness)
        if i%10==0:
            print(i,fitness_best_list[-1],"value:",value,"gbest",gbest)

    print("最优值是：%.5f" % fitness_best_list[-1])
    print("最优解是：",(gbest))


    #作图
    print(gbest)

    plt.figure()
    plt.xlim(0,100000)
    plt.ylim(0, 100000)
    plt.scatter(gbest[0:3*N-2:3],gbest[1:3*N-1:3])
    banjing=gbest[-N::1]

    for i in range(N):
        plt.scatter(np.linspace(gbest[3*i] - banjing[i], gbest[3*i] + banjing[i], 10000),
                    gbest[3*i+1] - np.sqrt(banjing[i] ** 2 - (
                            np.linspace(gbest[3*i] - banjing[i], gbest[3*i] + banjing[i], 10000) - gbest[3*i]) ** 2),color='r',s=0.001)
        plt.scatter(np.linspace(gbest[3 * i] - banjing[i], gbest[3 * i] + banjing[i], 10000),
                    gbest[3 * i + 1] + np.sqrt(banjing[i] ** 2 - (
                            np.linspace(gbest[3 * i] - banjing[i], gbest[3 * i] + banjing[i], 10000) - gbest[
                        3 * i]) ** 2),color='r',s=0.001)

    plt.show()

    X_best = gbest[np.newaxis, :].copy()
    loc_N_best = X_best[:, :3 * N].copy()
    P_n_best = X_best[:, 3 * N:4 * N].copy()
    B_n_best = X_best[:, 4 * N:5 * N].copy()
    r_n_best = X_best[:, 5 * N:6 * N].copy()
    shape1_best = (1, N, 3)
    shape2_best = (1, N)
    shape3_best = (1, N)
    shape4_best = (1, N)
    information = [loc_N_best.reshape(shape1_best), P_n_best.reshape(shape2_best), B_n_best.reshape(shape3_best),
                   r_n_best.reshape(shape4_best)]

    env.picture(information)
    plt.figure()
    plt.plot(fitness_best_list, c='r')
    #plt.title('迭代过程')
    plt.show()



if __name__ == '__main__':
    pos()
    #X = np.random.uniform(-2, 2, 10)
