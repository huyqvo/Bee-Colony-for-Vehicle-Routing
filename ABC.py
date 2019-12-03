from initialization import VRP
from searchSpace import SearchSpace

import numpy as np
import pandas as pd
import random

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
        print('Yeah 1')
        arr1 = np.flip(arr1)
    arr2 = np.copy(arr[s2:e2+1])
    if prob2 > 0.5:
        print('Yeah 2')
        arr2 = np.flip(arr2)

    print(s1, ' ', e1, ' ', s2, ' ', e2)
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

        return listOfProbs.sort(reverse=True)

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

        return selectionResults

    

# limit = 50n, m >= 3, (n+m) >= 7
n = int(input('Enter n:'))
m = int(input('Enter m:'))
k = int(input('Enter k:'))
VRPProb = VRP(n, m, k)

VRPProb.readData('D:\\University\\Nam 4 HK 1\\Soft computing\\DoAn_CK\\code\\data\\Problem_8.txt')
solList = VRPProb.initSols()

for sol in solList:
    print('[+] normal: ', sol)
    print('[+] swapReverse: ', swapReverse_neighborOps(sol))
