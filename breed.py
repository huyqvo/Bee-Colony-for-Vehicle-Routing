import numpy as np
import random
from math import sqrt

'''def breed(x, y):
    child = []
    childP1 = []
    childP2 = []

    # Choose random subsequence in x
    l = x.shape[0]
    while True:
        s = random.randint(0, l-4)
        if x[s] == 0 or (x[s+1] != 0 and x[s+2] != 0 and x[s+3] != 0):
            break
    while True:
        e = random.randint(s+3, l-1)
        if x[e] != 0 and x[e-1] != 0 and x[e-2] != 0 and x[e-3] != 0:
            break

    for i in range(s, e+1):
        childP1.append(x[i])
    nonzero_childP1 = list(filter(lambda a: a != 0, childP1))
    zero_n = len(childP1) - len(nonzero_childP1)

    childP2 = [item for item in y if item not in nonzero_childP1]
    indices = [i for i, x in enumerate(childP2) if x == 0]
    
    inds = random.sample(indices, k=zero_n)
    #print('inds: ', inds)
    for ind in sorted(inds, reverse=True):
        del childP2[ind]
        
    # Combine childP1 and childP2
    child = childP1.copy()
    print('s childP1: ', childP1)
    print('s childP2: ', childP2)
    if childP1[0] != 0:
        child = [0] + child
        childP2.remove(0)
    print('e childP1: ', childP1)
    print('e childP2: ', childP2)
    child += childP2
    indices = [i for i, x in enumerate(child) if x == 0]

    print('child: ', child)
    print('len(zero): ', len(indices))
    print()

    return np.asarray(child)'''
    
def calDist(i1, i2, infoList):
    '''
        Calculate distance between two customers having indices i1 and i2
    '''
    cust1 = infoList[i1]
    cust2 = infoList[i2]
    '''print(cust1)
    print(cust2)
    print((cust1[0] - cust2[0])**2)
    print((cust1[1] - cust2[1])**2)'''
    return sqrt((cust1[0] - cust2[0])**2 + (cust1[1] - cust2[1])**2)

def breed(x, y, infoList):
    child = []
    childP1 = []
    childP2 = []

    # Choose random subsequence in x
    indices = [i for i, val in enumerate(x) if val == 0]
    l = x.shape[0]
    veh = len(indices) # number of vehicle
    cust = l - veh # number of customer
    '''while True:
        s = random.randint(0, l-4)
        if x[s] == 0 or (x[s+1] != 0 and x[s+2] != 0 and x[s+3] != 0):
            break
    while True:
        e = random.randint(s+3, l-1)
        if x[e] != 0 and x[e-1] != 0 and x[e-2] != 0 and x[e-3] != 0:
            break'''
    
    s = random.randint(0, l-4)
    e = random.randint(s+3, l-1)

    for i in range(s, e+1):
        childP1.append(x[i])
    nonzero_childP1 = list(filter(lambda a: a != 0, childP1))
    #zero_n = len(childP1) - len(nonzero_childP1)

    childP2 = [item for item in y if item not in nonzero_childP1]
    nonzero_childP2 = list(filter(lambda a: a != 0, childP2))
        
    # Combine childP1 and childP2
    child = nonzero_childP1.copy()
    #child += nonzero_childP2.copy()
    for ele1 in nonzero_childP2:
        cur_l = len(nonzero_childP1)
        min_dist = 999999
        min_ind = -1
        for i in range(cur_l-1):
            prev_dist = calDist(ele1, nonzero_childP1[i], infoList)
            post_dist = calDist(ele1, nonzero_childP1[i+1], infoList)
            if prev_dist+post_dist < min_dist:
                min_dist = prev_dist + post_dist
                min_ind = i
        nonzero_childP1[min_ind+1:min_ind+1] = [ele1]
    child = nonzero_childP1.copy()


    child = [0] + child
    ran = int(cust/veh)
    s_ind = ran
    posList = []
    for i in range(veh - 1):
        pos = random.randint(s_ind, s_ind+int(ran/2)-1)
        posList.append(pos)
        s_ind += ran

    for pos in sorted(posList, reverse=True):
        child[pos:pos] = [0]

    #print('child: ', child)
    '''print('len(zero): ', len(indices))
    print()'''

    return np.asarray(child)

def popBreed(matingpool, infoList, eliteSize=2):
    """
    Perform breading to form new population
    """
    #print('[+] len(matingpool): ', len(matingpool))
    children = []
    length = len(matingpool) - eliteSize 
    pool = random.sample(matingpool, len(matingpool)) # sampling without replacement

    # Keep elitism indivs
    for i in range(0,eliteSize):
        #print('[+] i: ', i)
        children.append(pool[i])
    
    for i in range(0, length): # new children
        child = breed(pool[i], pool[len(pool)-i-1], infoList)
        children.append(child)

    return children # len(children) = len(matingpool)