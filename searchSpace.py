import numpy as np
from initialization import VRP

class SearchSpace:
    def __init__(self, n, m, k, c, alpha, theta, input_file): 
        '''
            n: so luong khach hang
            m: so luong phuong tien van chuyen 
            k: so luong solution trong moi iteration
            c: khoi luong moi xe co the mang
        '''
        self.alpha = float(alpha)
        self.theta = float(theta) # divided by
        self.c = c
        self.vrp = VRP(n, m, k)
        self.vrp.readData(input_file)

    def getViolationWeight(self, x):
        l = x.shape[0]
        # Capacity violation
        weight = 0
        weightList = []
        for i in range(1, l):               
            if x[i] == 0:
                weightList.append(weight)
                weight = 0
            weight += self.vrp.getWeight(x[i]) * 51 # 1 cwt = 50.8 kilogram
        weightList.append(weight)

        violationWeight = 0
        for w in weightList:
            if w - self.c > 0:
                violationWeight += w - self.c

        return violationWeight

    def calPathLengths(self, x): # x is a numpy array representing solution, !! this x has to have 0 at the end !!
        ret = []
        l = x.shape[0]
        dist = 0
        for i in range(0, l-1):
            if x[i] == 0 and i != 0:
                ret.append(dist)
                dist = 0
            dist += self.vrp.calDist(x[i], x[i+1])
        ret.append(dist)

        return ret

    def costFunc(self, x): # x is a numpy array representing solution
        # Distance
        dist = 0 # total distance of all vehicles
        l = x.shape[0]
        for i in range(l-1):
            '''if x[i+1] == 0:
                continue'''
            dist += self.vrp.calDist(x[i], x[i+1])
        dist += self.vrp.calDist(x[l-1], 0)

        violationWeight = self.getViolationWeight(x)

        return dist + self.alpha*violationWeight
        

    def updateAlpha(self, divOrMul):
        if divOrMul == True:
            self.alpha /= 1 + self.theta
        else:
            self.alpha *= 1 + self.theta