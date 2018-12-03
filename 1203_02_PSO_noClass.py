# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:28:28 2018
Particle Swarm Optimization (PSO)
REFER:
    http://edisonx.pixnet.net/blog/post/81640299-%5Bpso%5D-%E5%88%9D%E6%AD%A5---%E7%B2%92%E5%AD%90%E7%A7%BB%E5%8B%95%E6%BC%94%E7%AE%97%E6%B3%95%E7%B2%BE%E9%AB%93
    http://ccy.dd.ncu.edu.tw/~chen/resource/pso/pso.htm
    https://zh.wikipedia.org/wiki/%E7%B2%92%E5%AD%90%E7%BE%A4%E4%BC%98%E5%8C%96
PLT:
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(120)
TWO = 2

import time
print("START\n")
_startTime = time.time()
#%%
def MakeFigure(plotList, number, figSize=None):
#    print(plotList)
    x = plotList[:, 0]
    y = plotList[:, 1]
#    print("x,",x,"y,",y,sep="\n")
    fig = plt.figure()
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.plot(x, y, 'r*')
    
    plt.savefig("TMP/"+str(number)+".png")
#    plt.show()
    plt.close(fig)
    return
#%%
targetFunc = lambda x: 3*x[0]**2 - 2*x[0]*x[1] + 3*x[1]**2 - x[0] -x[1]
#
particleNum = 5 #M
x_max =  5
x_min = -5
x_range = x_max - x_min
v_max =  x_range /4.0
v_min = -x_range /4.0
#
k_time = 0 
k_time_max = 50
#
x_Position = np.zeros((particleNum, TWO, k_time_max))
x_Velocity = np.zeros((particleNum, TWO, k_time_max))
P_Best_x   = np.zeros((particleNum, TWO, 1)) # personal best
P_Best_fit = np.zeros((particleNum, 1)) # personal best
G_Best_x   = np.zeros((TWO)) # global best
G_Best_fit = np.zeros((1)) # global best

#
omaga = np.ones((particleNum, TWO))
c1 = np.ones((particleNum, TWO)) * 0.8
c2 = np.ones((particleNum, TWO)) * 0.2
#
boolFirst_P = np.array([True for i in range(particleNum)])
boolFirst_G = True
#%%
# 1. Initialize the swarm forming solution space 
#print((x_min + (x_max - x_min) * np.random.rand(particleNum, 1))[:, 0])
x_Position[:, :, 0] = (x_min + x_range * np.random.rand(particleNum, 2))
x_Velocity[:, :, 0] = (v_min + (v_max - v_min) * np.random.rand(particleNum, 2))

MakeFigure(x_Position[:, :, 0], 0)
for k in range(k_time_max-1):
    for i in range(particleNum):
# 2. Evaluate the fitness of each particle 
#        print(x_Position[i, :, k])
        tmpFitness = targetFunc(x_Position[i, :, k])
# 3. Update individual and global bests
        if (tmpFitness < P_Best_fit[i, 0]) or boolFirst_P[i]:
            boolFirst_P[i] = False
            P_Best_x  [i, :, 0] = x_Position[i, :, k]
            P_Best_fit[i, 0] = tmpFitness
        if (tmpFitness < G_Best_fit[0]) or boolFirst_G:
            boolFirst_G = False
            G_Best_x  [:] = x_Position[i, :, k]
            G_Best_fit[0] = tmpFitness
# 4. Update velocity and position of each particle 
        #
        x_Velocity[i, :, k+1] = omaga[i, :] * x_Velocity[i, :, k] \
                            + c1[i, :] * (P_Best_x[i, :, 0] - x_Position[i, :, k]) \
                            + c2[i, :] * (G_Best_x[:] - x_Position[i, :, k])
        #
        x_Position[i, :, k+1] = x_Position[i, :, k] + x_Velocity[i, :, k+1]
# 5. Go to Step 2, and repeat until termination criterion is Go to Step 2, and repeat until termination criterion is met.
    # 輸出存圖
    MakeFigure(x_Position[:, :, k+1], k+1)
#    break
#%%
print("global best:")
print("fitness:", G_Best_fit[0])
print("point:", G_Best_x)


print("\n\nEND", "It cost", time.time() - _startTime, "sec.")