from initialization import VRP
import numpy as np
import random

def swapReverse_neighborOps(arr, maxLen):
    while True:
        s1 = random.randint(0, arr.shape[0])
        l1 = random.randint(2, maxLen)
        if 0 in arr[s1:s1+l1]:
            break
    while True:
        while True:
            s2 = random.randint(0, arr.shape[0])
            if s2 > s1 + l1:
                break 
        l2 = random.randint(2, maxLen)
        if 0 in arr[s2:s2+l2]:
            break

    prob1 = np.random.uniform()
    prob2 = np.random.uniform()
    arr1 = np.copy(arr[s1:s1+l1])
    if prob1 > 0.5:
        arr1 = np.flip(arr1)
    arr2 = np.copy(arr[s2:s2+l2])
    if prob2 > 0.5:
        arr2 = np.flip(arr2)

    new_arr = np.empty(0)
    new_arr = np.concatenate((new_arr, np.copy(arr[0:s1])))
    new_arr = np.concatenate((new_arr, arr2))
    new_arr = np.concatenate((new_arr, arr[s1+l1:s2]))
    new_arr = np.concatenate((new_arr, arr[s2+l2:]))

    return new_arr
    


# limit = 50n
n = int(input('Enter n:'))
m = int(input('Enter m:'))
k = int(input('Enter k:'))
VRPProb = VRP(n, m, k)

VRPProb.readData('D:\\University\\Nam 4 HK 1\\Soft computing\\DoAn_CK\\code\\data\\Problem_8.txt')
solList = VRPProb.initSols()

for sol in solList:
    print(sol)
