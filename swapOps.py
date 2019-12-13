import random
import numpy as np

CONST_REVERSE = 0
CONST_SWAP = 1
CONST_SWAP_REVERSE = 2

def pick_random_op():
    """
    Pick randomly 1 operator among 3
    """
    randnum = random.randint(0, 2)
    if randnum == CONST_REVERSE:
        return reverse_subseqence
    elif randnum == CONST_SWAP:
        return swap_ops
    else:
        return swapReverse_neighborOps
    
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
        # extend the first subsequence
        res.extend(arr[first_index:len(arr)])
        # extend the remaining subsequence between the first and next depot
        res.extend(arr[second_index+1:depot_indices[next_depot_index]])
        # extend the middle subsequence
        res.extend(arr[depot_indices[next_depot_index]:first_index])
        # extend the second subsequence at the tail
        res.extend(np.flip(arr[depot_indices[random_depot_index]+1:second_index+1]))

        arr = np.array(res)
    # if it is the last depot in depot
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
    print(arr)
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

# for i in range(1000):
#     res = reverse_subseqence(np.array([0, 4, 1, 0, 2, 7, 5, 0, 3, 6]))
#     for j in range(len(res)-1):
#         if res[j] == res[j + 1]:
#             print(True)