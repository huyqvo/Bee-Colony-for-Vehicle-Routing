'''
    Standard benchmark from Christofides (1969): An Algorithm for the Vehicle-dispatching problem

    - Problem 8: 50 customer
    - Problem 9: 75 customer

    All the problems above only have distance and demand (weight) constraint, don't have time constraint. Depot and customer location are represented in Euclidean Co-ordinates.
    Depot is at (30, 40)
'''

import random
from math import sqrt
import numpy as np

class VRP:
    def __init__(self, n, m, k):
        self.n = n # The number of customers
        self.m = m # The number of vehicles
        self.k = k # The number of initial solutions
        self.infoList = [] # List of customers'information: list of tuple having 3 elements (x, y, q)

    def getInfoList(self):
        return self.infoList

    def readData(self, filepath):
        f = open(filepath, 'r')
        lines = f.readlines()

        self.infoList.append((30, 40, 0)) # !! Depot information, default to (30, 40) !!
        for line in lines:
            chars = line[:-1].split('\t')
            #print(chars)
            info = (int(chars[1]), int(chars[2]), int(chars[3]))
            self.infoList.append(info)

        return self.infoList.copy()

    def getWeight(self, i1):
        '''
            Get weight of vehicle at index i1
        '''
        return self.infoList[i1][2]

    def calDist(self, i1, i2):
        '''
            Calculate distance between two customers having indices i1 and i2
        '''
        cust1 = self.infoList[i1]
        cust2 = self.infoList[i2]
        '''print(cust1)
        print(cust2)
        print((cust1[0] - cust2[0])**2)
        print((cust1[1] - cust2[1])**2)'''
        return sqrt((cust1[0] - cust2[0])**2 + (cust1[1] - cust2[1])**2)

    def createRandomSol(self):
        vehicleList = [] # List of vehicle routes, #vehicles = #routes. Don't have 0 at start and end. For saving index of customer
        distList = [] # List of list of distances between 2 adjacent customer in a route
        totalDistList = [] # List of distances of routes
        for i in range(self.m):
            vehicleList.append([0])
            distList.append([])
            totalDistList.append(0)
        #print(vehicleList)

        remainedCust = list(range(0, self.n))                           
        for i in range(self.n):
            custIndex = random.choice(remainedCust) # Chosen customer index
            remainedCust.remove(custIndex)

            # minimize distance
            minDist = 999999
            minIndex = self.m # storing index of vehicle route
            for j in range(self.m):
                lastCustIndex = vehicleList[j][len(vehicleList[j]) - 1]
                curDist = self.calDist(custIndex, lastCustIndex)
                if curDist < minDist:
                    minDist = curDist
                    minIndex = j

            vehicleList[minIndex].append(custIndex+1)

        # Make a representation vector
        sol = []
        for i in range(self.m):
            sol.append(0)
            cust_len = len(vehicleList[i])
            for j in range(1, cust_len):
                sol.append(vehicleList[i][j])
        sol = np.asarray(sol)

        return sol


    def initSols(self):
        solList = [] # list of representation vectors
        for k in range(self.k):
            sol = self.createRandomSol()
            solList.append(sol)

        return solList
            
    