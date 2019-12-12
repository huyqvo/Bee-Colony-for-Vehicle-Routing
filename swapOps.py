import random
import numpy as np

def swap_ops(arr): 
    '''
        Random swap 2 elements in representation vector, for step (e)
    '''
    l = arr.shape[0]
    while True:
        ele1 = random.randint(1, l-1)
        if arr[ele1] != 0:
            break
    while True:
        ele2 = random.randint(1, l-1)
        if arr[ele2] != 0 and ele2 != ele1:
            break

    ret = np.copy(arr)
    ret[ele1], ret[ele2] = ret[ele2], ret[ele1]

    return ret


def swapReverse_neighborOps(arr, m):
    '''
        Swap reverse of 2 subsequences
    '''
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
