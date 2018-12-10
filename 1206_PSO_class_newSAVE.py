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
gif:
    https://ezgif.com/maker
"""
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
def MakeFigure(plotList, number, figSize=None, imgFolder = "TMP_s1200_250"):
#    print(plotList)
    x = plotList[:, 0]
    y = plotList[:, 1]
#    print("x,",x,"y,",y,sep="\n")
    fig = plt.figure()
    
    plt.title(str(number))
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.plot(x, y, 'r*')
    
    
    plt.savefig(imgFolder + "/" +str(number)+".png")
#    plt.show()
    plt.close(fig)
    return

#%%
class PSO:
    def __init__(self, seed = None):
        if seed is None :
            self.randSeed = None
        else: 
            self.randSeed = seed
            np.random.seed(seed)
        TWO = 2
        #
        self.targetFunc = lambda x: 3*x[0]**2 - 2*x[0]*x[1] + 3*x[1]**2 - x[0] -x[1]
        #
        self.particleNum = 5 #M
        self.x_max =  5
        self.x_min = -5
        self.x_range = self.x_max - self.x_min
        self.v_max =  self.x_range /4.0
        self.v_min = -self.x_range /4.0
        #
        self.k_time = 0 
        self.k_time_max = 250
        #
        self.x_Position = np.zeros((self.particleNum, TWO, self.k_time_max))
        self.x_Velocity = np.zeros((self.particleNum, TWO, self.k_time_max))
        self.P_Best_x   = np.zeros((self.particleNum, TWO, 1)) # personal best
        self.P_Best_fit = np.zeros((self.particleNum, 1)) # personal best
        self.G_Best_x   = np.zeros((TWO)) # global best
        self.G_Best_fit = np.zeros((1)) # global best
        
        #
        self.omaga = np.ones((self.particleNum, TWO)) * 1
        self.c1 = np.ones((self.particleNum, TWO))    * 0.8
        self.c2 = np.ones((self.particleNum, TWO))    * 0.2
        #
        self.boolFirst_P = np.array([True for i in range(self.particleNum)])
        self.boolFirst_G = True
        # 
        self.folderName = "output_s"+str(self.randSeed)+"_"+str(self.k_time_max)
        if os.path.exists(self.folderName) is False:
            os.mkdir(self.folderName)
        return
    
    def FLOW(self):
        # 1. Initialize the swarm forming solution space 
        #print((x_min + (x_max - x_min) * np.random.rand(particleNum, 1))[:, 0])
        self.x_Position[:, :, 0] = (self.x_min + self.x_range * np.random.rand(self.particleNum, 2))
        self.x_Velocity[:, :, 0] = (self.v_min + (self.v_max - self.v_min) * np.random.rand(self.particleNum, 2))
        MakeFigure(self.P_Best_x[:, :, 0], 0, imgFolder = self.folderName)
        
        for k in range(self.k_time_max-1):
            for i in range(self.particleNum):
        # 2. Evaluate the fitness of each particle 
        #        print(x_Position[i, :, k])
                tmpFitness = self.targetFunc(self.x_Position[i, :, k])
        # 3. Update individual and global bests
                if (tmpFitness < self.P_Best_fit[i, 0]) or self.boolFirst_P[i]:
                    self.boolFirst_P[i]   = False
                    self.P_Best_x  [i, :, 0] = self.x_Position[i, :, k]
                    self.P_Best_fit[i, 0] = tmpFitness
                if (tmpFitness < self.G_Best_fit[0]) or self.boolFirst_G:
                    self.boolFirst_G   = False
                    self.G_Best_x  [:] = self.x_Position[i, :, k]
                    self.G_Best_fit[0] = tmpFitness
        # 4. Update velocity and position of each particle 
                #
                self.x_Velocity[i, :, k+1] = self.omaga[i, :] * self.x_Velocity[i, :, k] \
                                    + self.c1[i, :] * np.random.rand(1) * (self.P_Best_x[i, :, 0] - self.x_Position[i, :, k]) \
                                    + self.c2[i, :] * np.random.rand(1) * (self.G_Best_x[:] - self.x_Position[i, :, k])
                #
                self.x_Position[i, :, k+1] = self.x_Position[i, :, k] + self.x_Velocity[i, :, k+1]
        # 5. Go to Step 2, and repeat until termination criterion is Go to Step 2, and repeat until termination criterion is met.
            MakeFigure(self.P_Best_x[:, :, 0], k+1, imgFolder = self.folderName)
        return
    
    def ShowResault(self):
        print("global best:")
        print("fitness:", self.G_Best_fit[0])
        print("point:",   self.G_Best_x)
        return
if __name__ == "__main__":
    import time
    print("START\n")
    _startTime = time.time()
    
    pso = PSO(seed = 1200)
    pso.FLOW()
    pso.ShowResault()
    
    end_position = pso.x_Position
    end_velocity = pso.x_Velocity
    
    print("\n\nEND", "It cost", time.time() - _startTime, "sec.")