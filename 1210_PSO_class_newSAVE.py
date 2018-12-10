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
    https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    http://imageio.github.io/
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
#%%
def MakeFigure(plotList, number, figSize=None, imgFolder = "TMP"):
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

def MakeForderImgToGif(imgFolder):
#    print(imgFolder.rsplit('/',1))
    # name sort
    imgNameList = np.array(os.listdir(imgFolder))
    imgNameList = imgNameList[np.argsort([int(na.split(".")[0]) for na in imgNameList ], axis = 0)]
    # 
    with imageio.get_writer(imgFolder+ '.gif', mode='I') as writer:
        for filename in imgNameList:
            image = imageio.imread(imgFolder + "/" + filename)
            writer.append_data(image)
    return

def CleanGifFolder(imgFolder):
    imgNameList = np.array(os.listdir(imgFolder))
    for na in imgNameList:
        os.remove(imgFolder + "/" + na)
    os.removedirs(imgFolder)
    return
#%%
class PSO:
    def __init__(self, particleNum = 15, time_max = 50, seed = None, boolSaveFig = True):
        if seed is None :
            self.randSeed = None
        else: 
            self.randSeed = seed
            np.random.seed(seed)
        TWO = 2
        #
        self.__boolSaveFig = boolSaveFig
        self.targetFunc = lambda x: 3*x[0]**2 - 2*x[0]*x[1] + 3*x[1]**2 - x[0] -x[1]
        #
        self.particleNum = particleNum # M
        self.x_max =  5
        self.x_min = -5
        self.x_range = self.x_max - self.x_min
        self.v_max =  self.x_range /4.0
        self.v_min = -self.x_range /4.0
        #
#        self.k_time = 0 
        self.k_time_max = time_max
        # 
        self.x_Position = np.zeros((self.particleNum, TWO, self.k_time_max))
        self.x_Velocity = np.zeros((self.particleNum, TWO, self.k_time_max))
        self.P_Best_x   = np.zeros((self.particleNum, TWO, 1)) # personal best
        self.P_Best_fit = np.zeros((self.particleNum, 1)) # personal best
        self.G_Best_x   = np.zeros((TWO)) # global best
        self.G_Best_fit = np.zeros((1)) # global best
        # 變異係數
        self.omaga = np.ones((self.particleNum, TWO)) * 1   #動量
        self.c1 = np.ones((self.particleNum, TWO))    * 0.8
        self.c2 = np.ones((self.particleNum, TWO))    * 0.2
        # 第一項執行控制
        self.boolFirst_P = np.array([True for i in range(self.particleNum)])
        self.boolFirst_G = True
        # 
        self.folderName = "output_s"+str(self.randSeed)+"_t"+str(self.k_time_max)+"_p"+str(self.particleNum)
        if os.path.exists(self.folderName) is False and self.__boolSaveFig:
            os.mkdir(self.folderName)
        return
    
    def FLOW(self):
        # 1. Initialize the swarm forming solution space 
        #print((x_min + (x_max - x_min) * np.random.rand(particleNum, 1))[:, 0])
        self.x_Position[:, :, 0] = (self.x_min + self.x_range * np.random.rand(self.particleNum, 2))
        self.x_Velocity[:, :, 0] = (self.v_min + (self.v_max - self.v_min) * np.random.rand(self.particleNum, 2))
        if self.__boolSaveFig:
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
            if self.__boolSaveFig:
                MakeFigure(self.P_Best_x[:, :, 0], k+1, imgFolder = self.folderName)
        # 6. 動量(omaga)下降
            if self.omaga[0, 0] > 0.01:
                self.omaga -= 0.01 
        return
    
    def ShowResault(self):
        print("global best:")
        print("fitness:", self.G_Best_fit[0])
        print("point:",   self.G_Best_x)
        # 輸出 gif
        if self.__boolSaveFig:
            MakeForderImgToGif(self.folderName)
        return
if __name__ == "__main__":
    import time
    print("START\n")
    _startTime = time.time()
    
    pso = PSO(particleNum = 50, time_max = 250, seed = 1200, boolSaveFig = True)
    pso.FLOW()
    pso.ShowResault()
    
    end_position = pso.x_Position
    end_velocity = pso.x_Velocity
    
    if True: #清空 gif 輸出
        CleanGifFolder(pso.folderName)
    
    print("\n\nEND", "It cost", time.time() - _startTime, "sec.")