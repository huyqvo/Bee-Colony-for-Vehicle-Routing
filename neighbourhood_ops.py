import random
import numpy as np

CONST_SWAP = 0
CONST_SWAP_REVERSE = 1
CONST_REVERSE = 2

def pick_random_op():
    """
    Pick randomly 1 operator among 3
    """
    randnum = random.randint(0, 1)
    # randnum = random.choice([0, 2])
    if randnum == CONST_REVERSE:
        return reverse_subseqence
    elif randnum == CONST_SWAP:
        return swap_ops
    else:
        return swapReverse_neighborOps

def check_consecutive_dipot(arr):
    """
    Check whether arr has 2 consecutive dipot
    """
    for j in range(len(arr)-1):
        if arr[j] == arr[j + 1]:
            return True
    return False    
    
def reverse_subseqence(arr):
    """
    Reverse a subsequence (Modified version)
    """
    # Find depot indices (Paper dont do this)
    depot_indices = np.where(arr == 0)[0]
    # print(depot_indices)
    # Pick randomly 1 depot index in depot_indices
    random_depot_index = random.randint(0, len(depot_indices) - 1)
    # print(random_depot_index)
    # Find the previous and next depot indices of random_depot_index in depot_indices
    prev_depot_index, next_depot_index = random_depot_index - 1, random_depot_index + 1
    # print(prev_depot_index, next_depot_index)

    # If it is the first depot in arr, then we need to wrap the arr vector to reverse the sequence
    if random_depot_index == 0:
        prev_depot_index = len(depot_indices) - 1
        # print(prev_depot_index)
        first_index = random.randint(depot_indices[prev_depot_index]+1, len(arr)-1)
        second_index = random.randint(depot_indices[random_depot_index]+1, depot_indices[next_depot_index]-1)
        # print(first_index, second_index)
        res = []
        # First postion doesnt change
        res.append(0)
        # extend the first subsequence from first_index to the end of arr (not reverse)
        res.extend(arr[first_index:len(arr)])
        # extend the remaining subsequence between the first and next depot
        res.extend(arr[second_index+1:depot_indices[next_depot_index]])
        # extend the middle subsequence
        res.extend(arr[depot_indices[next_depot_index]:first_index])
        # extend the reversed second subsequence at the tail
        res.extend(np.flip(arr[depot_indices[random_depot_index]+1:second_index+1]))

        arr = np.array(res)
    # if it is the last depot in depot_indices
    elif random_depot_index == len(depot_indices) - 1:
        next_depot_pos = len(arr) # position in arr
        start_index = random.randint(depot_indices[prev_depot_index]+1, depot_indices[random_depot_index]-1)
        end_index = random.randint(depot_indices[random_depot_index]+1, next_depot_pos-1)
        arr[start_index:end_index+1] = np.flip(arr[start_index:end_index+1])                
    # Otherwise
    else:
        start_index = random.randint(depot_indices[prev_depot_index]+1, depot_indices[random_depot_index]-1)
        end_index = random.randint(depot_indices[random_depot_index]+1, depot_indices[next_depot_index]-1)
        arr[start_index:end_index+1] = np.flip(arr[start_index:end_index+1])
    
    return arr

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
    # print(arr)
    l = arr.shape[0]
    zero_pos = np.where(arr==0)[0]
    zero_pos = zero_pos[1:]
    zero_pos = zero_pos.reshape(m-1)
    while True:
        zero_n = np.random.choice(zero_pos, 2, replace=False)
        if abs(zero_n[1] - zero_n[0]) > 4 and zero_n[0] != l-2 and zero_n[1] != l-2:
            break
    if zero_n[0] > zero_n[1]:
        zero_n[0], zero_n[1] = zero_n[1], zero_n[0]

    pos1 = np.where(zero_pos == zero_n[0])[0]
    pos2 = np.where(zero_pos == zero_n[1])[0]

    s1 = e1 = s2 = e2 = 0
    if pos1 == 0:
        s1 = 1
    else:
        s1 = int(zero_pos[pos1 - 1])+1
    if pos1 == pos2 - 1:
        dist = zero_n[1] - zero_n[0]
        e1 = zero_n[0] + int(dist/2)
        s2 = e1 + 1
    else:
        e1 = int(zero_pos[pos1+1])-1
        s2 = int(zero_pos[pos2-1])+1
    if pos2 < int(zero_pos.shape[0])-1:
        e2 = int(zero_pos[pos2+1])-1
    else:
        e2 = arr.shape[0]-1
    '''print(zero_pos)'''
    '''print(arr)
    print(zero_n)
    print(s1, ' ', e1, ' ', s2, ' ', e2)'''

    #print('s1, zero_n[0]-2: ', s1, zero_n[0]-2)
    try:
        start1 = random.randint(s1, zero_n[0]-2)
    except:
        print(arr)
        print('s1, zero_n[0]-2: ', s1, zero_n[0]-2)
    #print('zero_n[0]+1, e1: ', zero_n[0]+2, e1)
    end1 = random.randint(zero_n[0]+2, e1)
    #print('s2, zero_n[1]-1: ', s2, zero_n[1]-2)
    start2 = random.randint(s2, zero_n[1]-2)
    #print('zero_n[1]+1, e2: ', zero_n[1]+2, e2)
    try:
        end2 = random.randint(zero_n[1]+2, e2)
    except:
        print(arr)
        print('zero_n[1]+2, e2: ', zero_n[1]+2, e2)

    s1 = start1
    e1 = end1
    s2 = start2
    e2 = end2

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
    
    '''check1 = -1
    check2 = -1
    l = new_arr.shape[0]
    for i in range(l-1):
        if new_arr[i] == new_arr[i+1]:
            check1 = i
    if new_arr[l-1] == 0:
        check2 = 1
    if check1 != -1:
        print()
        print('check1: ', s1, ' ', e1, ' ', s2, ' ', e2)
        print(arr)
        print(new_arr)
    if check2 != -1:
        print()
        print('check2: ', s1, ' ', e1, ' ', s2, ' ', e2)
        print(arr)
        print(new_arr)'''

    '''if new_arr[2] == 0:
        return arr'''
    return new_arr

'''def swapReverse_neighborOps(arr, m):
    # print(arr)
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

    # for j in range(len(new_arr)-1):
    #     if new_arr[j] == new_arr[j + 1]:
    #         print(arr)
    #         print(new_arr)
    #         print(new_arr[j], new_arr[j + 1])
    #         print(True)
    #         break

    return new_arr'''
# res = np.array([0,  5, 49, 30, 10, 39, 33, 45, 15, 44, 42, 17,  0, 23, 24, 18,  4, 41, 40, 19, 37, 12,  0, 27,
#                 8, 26, 31, 28,  3, 36, 35, 20, 22, 32,  0, 47, 25, 43,  7, 48,  1,  2, 11,  0,  6, 14, 13, 46, 16, 21, 29, 50, 34,  9, 38])
# for i in range(100000):
#     res = reverse_subseqence(res)
#     for j in range(len(res)-1):
#         if res[j] == res[j + 1]:
#             print(True)