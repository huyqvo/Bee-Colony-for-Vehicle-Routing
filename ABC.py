from initialization import VRP
from searchSpace import SearchSpace

import numpy as np
import pandas as pd
import random
import operator

'''def swapReverse_neighborOps(arr):
    while True:
        s1 = random.randint(0, arr.shape[0] - 4)
        e1 = random.randint(s1+1, arr.shape[0] - 3)                                                                                                                      
        if 0 in arr[s1:e1+1]:
            break
    while True:
        s2 = random.randint(e1+1, arr.shape[0] - 2)
        e2 = random.randint(s2+1, arr.shape[0] - 1)
        if 0 in arr[s2:e2+1]:
            break

    prob1 = np.random.uniform()
    prob2 = np.random.uniform()
    arr1 = np.copy(arr[s1:e1+1])
    if prob1 > 0.5:
        arr1 = np.flip(arr1)
    arr2 = np.copy(arr[s2:e2+1])
    if prob2 > 0.5:
        arr2 = np.flip(arr2)
    
    print(arr1, ' ', arr2)

    new_arr = np.empty(0)
    new_arr = np.concatenate((new_arr, np.copy(arr[0:s1])))
    new_arr = np.concatenate((new_arr, arr2))
    new_arr = np.concatenate((new_arr, arr[e1+1:s2]))
    new_arr = np.concatenate((new_arr, arr1))
    new_arr = np.concatenate((new_arr, arr[e2+1:]))

    return new_arr'''

def swapReverse_neighborOps(arr):
    zero_pos = np.where(arr==0)[0]
    zero_pos = zero_pos[1:]
    zero_pos = zero_pos.reshape(m-1)
    zero_n = np.random.choice(zero_pos, 2, replace=False)
    if zero_n[0] > zero_n[1]:
        zero_n[0], zero_n[1] = zero_n[1], zero_n[0]

    pos1 = np.where(zero_pos == zero_n[0])[0]
    pos2 = np.where(zero_pos == zero_n[1])[0]

    # First subsequence
    s = 2
    if pos1-1 >= 0:
        s = zero_pos[pos1-1] + 1
    e = 2
    if zero_n[0] - 1 >= 2:
        e = zero_n[0] - 1
    s1 = random.randint(s, e)
    e1 = random.randint(zero_n[0]+1, zero_pos[pos1+1]-1)

    # Second subsequence
    s = zero_pos[pos2-1]+1
    if s <= e1:     
        s = e1+1
    e = s
    if zero_n[1] - 1 >= s:
        e = zero_n[1] - 1
    s2 = random.randint(s, e)
    e = arr.shape[0]-1
    if pos2+1 < zero_pos.shape[0]:
        e = zero_pos[pos2+1] - 1
    e2 = random.randint(zero_n[1]+1, e)

    # copy 
    prob1 = np.random.uniform()
    prob2 = np.random.uniform()
    arr1 = np.copy(arr[s1:e1+1])
    if prob1 > 0.5:
        #print('Yeah 1')
        arr1 = np.flip(arr1)
    arr2 = np.copy(arr[s2:e2+1])
    if prob2 > 0.5:
        #print('Yeah 2')
        arr2 = np.flip(arr2)

    # swap
    #print(s1, ' ', e1, ' ', s2, ' ', e2)
    new_arr = np.empty(0, dtype=int)
    new_arr = np.concatenate((new_arr, np.copy(arr[0:s1])))
    new_arr = np.concatenate((new_arr, arr2))
    new_arr = np.concatenate((new_arr, np.copy(arr[e1+1:s2])))
    new_arr = np.concatenate((new_arr, arr1))
    new_arr = np.concatenate((new_arr, np.copy(arr[e2+1:])))

    return new_arr


class ABC:
    def __init__(self, n, m, k, c, alpha, theta, employedBees, onlookers): # Is the number of employedBees equal k?
        # The number of employed bees and the number of onlookers are set to be equal
        # the number of food sources (set to 25 in the paper)
        self.k = k
        self.vrp = VRP(n,m,k)
        self.searchSpace = SearchSpace(n,m,k,c,alpha,theta)
        self.employedBees = employedBees
        self.onlookers = onlookers
        #self.cumulated_f = 0 # cumulated f at iteration t
        self.listOfFoodSources = []

    def calFitness(self, foodSource):
        cost = self.searchSpace.costFunc(foodSource)
        return float(1.0/cost)

    def probOfFoodSources(self):
        '''
            All of the probability of choosing a food source x
        '''
        listOfProbs = []
        cumulated_f = 0
        for i in range(self.k):
            cost = self.searchSpace.costFunc(self.listOfFoodSources[i])
            f = float(1.0/cost)
            listOfProbs.append(f)
            cumulated_f += f
        listOfProbs[:] = [x/cumulated_f for x in listOfProbs]

        dictOfProbs = {}
        for (i, prob) in enumerate(listOfProbs):
            dictOfProbs[i] = prob

        #return listOfProbs.sort(reverse=True)
        return sorted(dictOfProbs.items(), key=operator.itemgetter(1), reverse=True)

    def rouletteWheel(self, listOfProbs):
        selectionResults = [] # list of indices of food sources for corresponding onlookers
        df = pd.DataFrame(np.array(listOfProbs), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum() # cumulative sum
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

        for i in range(self.k):
            pick = 100*random.random()
            for j in range(self.k):
                if pick <= df.iat[j, 3]:
                    selectionResults.append(j)
                    break

        return selectionResults # list of onlookers' food source

    def process(self, maxIteration, maxLimit):
        #self.vrp.readData('./data/problem_8.txt')
        self.vrp.readData('D:\\Github\\Bee-Colony-for-Vehicle-Routing\\data\\problem_8.txt')
        self.listOfFoodSources = self.vrp.initSols()
        listOfProbs = self.probOfFoodSources() # Not fitness but probability
        limits = [0] * self.k
        for itera in range(maxIteration):
            # (a)
            for (i, foodSource) in enumerate(self.listOfFoodSources):
                # Apply a neigborhood operator
                x_tilde = swapReverse_neighborOps(foodSource)
                old_fit = self.calFitness(foodSource)
                new_fit = self.calFitness(x_tilde)
                # Replace
                if new_fit > old_fit:
                    self.listOfFoodSources[i] = x_tilde
            
            # (b)
            G = [[]] * self.k # list of neighbor sets of foodsource i 

            # (c)
            selection = self.rouletteWheel(listOfProbs)
            for index in selection:
                x_tilde = swapReverse_neighborOps(self.listOfFoodSources[index])
                print(type(index))
                #x_tilde = list(x_tilde)
                #if G[index].get(x_tilde) == None:
                G[index].append(x_tilde)

            # (d)
            for (i, foodSource) in enumerate(self.listOfFoodSources):
                if len(G[i]) != 0:
                    maxFitness = -1
                    maxNeighbor = -1
                    for neighbor in G[i]: # neighbor is a numpy array of solution representation
                        fit = self.calFitness(neighbor)
                        if fit > maxFitness:
                            maxFitness = fit
                            maxNeighbor = neighbor
                    if maxFitness > self.calFitness(foodSource):
                        self.listOfFoodSources[i] = maxNeighbor
                        limits[i] = 0
                    else:
                        limits[i] += 1

            # (e)
            for (i, foodSource) in enumerate(self.listOfFoodSources):
                if limits[i] == maxLimit:
                    self.listOfFoodSources[i] = self.vrp.createRandomSol()
                    limits[i] = 0 # Paper don't have this?

            # Update alpha
            cnt = 0
            for (i, foodSource) in enumerate(self.listOfFoodSources):
                if self.searchSpace.getViolationWeight(foodSource) == 0:
                    cnt += 1
            if cnt > self.k/2:
                self.searchSpace.updateAlpha(True) # divide
            else:
                self.searchSpace.updateAlpha(False) # multiply
            

        return self.listOfFoodSources

# limit = 50n, m >= 3, (n+m) >= 7

n = int(input('Enter n:'))
m = int(input('Enter m:'))
k = int(input('Enter k:')) # k = 25
c = int(input('Enter c:'))
alpha = 0.1 # according to paper
theta = 0.001 # according to the paper
employedBees = k
onlookers = k
#VRPProb = VRP(n, m, k)
abc = ABC(n,m,k,c,alpha,theta,employedBees,onlookers)

listOfFoodSources = abc.process(25, 5)

for foodSource in listOfFoodSources:
    print(foodSource)
'''VRPProb.readData('D:\\University\\Nam 4 HK 1\\Soft computing\\DoAn_CK\\code\\data\\Problem_8.txt')
solList = VRPProb.initSols()

for sol in solList:
    print('[+] normal: ', sol)
    print('[+] swapReverse: ', swapReverse_neighborOps(sol))'''
