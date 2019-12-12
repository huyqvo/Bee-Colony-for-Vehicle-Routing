from initialization import VRP
from searchSpace import SearchSpace

import numpy as np
import pandas as pd
import random
import operator

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

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

def swap_ops(arr): 
    '''
        Random swap 2 elements in representation vector
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
    def __init__(self, n, m, k, c, alpha, theta, employedBees, onlookers, input_file): # Is the number of employedBees equal k?
        # The number of employed bees and the number of onlookers are set to be equal
        # the number of food sources (set to 25 in the paper)
        self.k = k
        self.vrp = VRP(n,m,k)
        self.searchSpace = SearchSpace(n,m,k,c,alpha,theta, input_file)
        self.employedBees = employedBees
        self.onlookers = onlookers
        #self.cumulated_f = 0 # cumulated f at iteration t
        self.listOfFoodSources = []

    def visualize(self, solution):
        infoList = self.vrp.getInfoList()
        solution = np.append(solution, 0)

        x = []
        y = []
        w = []

        for info in infoList:
            x.append(info[0])
            y.append(info[1])
            w.append(info[2])
            plt.text(info[0], info[1]+1, str(info[2]), family="serif")

        plt.plot(x, y, 'ro')
        plt.plot(30, 40, 'mD', markersize=15)

        l = solution.shape[0]
        # colo = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        colo_index = -1

        color_palette = "rainbow" # More colormap instances: https://matplotlib.org/gallery/color/colormap_reference.html
        palette_samples = self.vrp.m # Number of palette samples are equal to #vehicles

        try:
            colors = np.array(plt.get_cmap(color_palette).colors)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples)))[:, -2::-1].tolist()
        
        for i in range(l-1):
            if solution[i] == 0:
                colo_index += 1
        
            x1 = [infoList[solution[i]][0], infoList[solution[i+1]][0]]
            y1 = [infoList[solution[i]][1], infoList[solution[i+1]][1]]
            plt.plot(x1, y1, color=tuple(colors[colo_index % len(colors)]), linestyle='-')
        
        #plt.plot(30, 40, color=colo[colo_index], linestyle='-')

        plt.axis([0,80,0,80])
        
        # ------ Visualize distance on legend ------

        # colo = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'whit
        listOfDists = self.searchSpace.calPathLengths(solution)
        vehicle_n = len(listOfDists)
        listOfPatches = []

        for i in range(vehicle_n):
            patch = mpatches.Patch(color=tuple(colors[i % len(colors)]), label='Path length = '+str(listOfDists[i]))
            listOfPatches.append(patch)

        '''red_patch = mpatches.Patch(color='red', label='The red data')
        blue_patch = mpatches.Patch(color='blue', label='The blue data')
        green_patch = mpatches.Patch(color='green', label='The green data')
        plt.legend(handles=[red_patch, blue_patch, green_patch])'''

        plt.legend(handles=listOfPatches)

        plt.show()

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
        #return sorted(dictOfProbs.items(), key=operator.itemgetter(1), reverse=True)
        return dictOfProbs

    def rouletteWheel(self, listOfProbs):
        sorted_ind = sorted(listOfProbs, key=listOfProbs.get, reverse = True)
        #print(sorted_ind)
        listOfProbs = sorted(listOfProbs.items(), key=operator.itemgetter(1), reverse=True)
        selectionResults = [] # list of indices of food sources for corresponding onlookers
        df = pd.DataFrame(np.array(listOfProbs), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum() # cumulative sum
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

        for i in range(2):
            selectionResults.append(i)
        for i in range(self.k-2):
            pick = 100*random.random()
            for j in range(self.k-1):
                if pick <= df.iat[j, 3] and pick > df.iat[j+1,3]:
                    selectionResults.append(sorted_ind[j])
                    break

        return selectionResults # list of onlookers' food source

    def process(self, maxIteration, maxLimit, input_file): 
        # Assume the number of employed bees is equal to the number of food sources
        #self.vrp.readData('./data/problem_8.txt')
        self.vrp.readData(input_file)
        self.listOfFoodSources = self.vrp.initSols()
        print('[+] Initial solution')
        for foodsource in self.listOfFoodSources:
            print(foodsource)
        print('[+] Final solution')
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
                    #print('yeah1')
                    self.listOfFoodSources[i] = x_tilde
            
            # (b)
            G = [[]] * self.k # list of neighbor sets of foodsource i 

            # (c)
            listOfProbs = self.probOfFoodSources() # Not fitness but probability
            selection = self.rouletteWheel(listOfProbs)
            for index in selection:
                x_tilde = swapReverse_neighborOps(self.listOfFoodSources[index])
                #print(type(index))
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
                        #print('yeah2')
                        self.listOfFoodSources[i] = maxNeighbor
                        limits[i] = 0
                    else:
                        limits[i] += 1

            '''for (i,foodsource) in enumerate(self.listOfFoodSources):
                print('[+] fitness ' + str(i) + ': ', self.calFitness(foodsource))'''
            '''for (i, foodsource) in enumerate(self.listOfFoodSources):
                print('[+] foodsource ' + str(i) + ': ', foodsource)'''
            #print()

            # (e)
            for (i, foodSource) in enumerate(self.listOfFoodSources):       
                if limits[i] == maxLimit:
                    muta_vec = swap_ops(self.listOfFoodSources[i])
                    old_fit = self.calFitness(self.listOfFoodSources[i])
                    new_fit = self.calFitness(muta_vec)
                    if new_fit > old_fit:
                        self.listOfFoodSources[i] = muta_vec
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

def ParseArguments():
    parser = argparse.ArgumentParser(description="Bee colony algorithm for Vehicle Routing Problem")
    parser.add_argument('-f', "--input_file", type=str,
                        help="Input file")
    parser.add_argument('-n', "--num_customers", type=int, 
                        default=50, help="Number of customers")
    parser.add_argument('-m', "--num_vehicles", type=int, 
                        default=3, help="Number of vehicles")
    parser.add_argument('-k', "--num_solutions", type=int, 
                        default=25, help="Number of solution (food sources)")
    parser.add_argument('-c', "--capacity", type=float, 
                        default=1000.0, help="Vehicle capacity")
    parser.add_argument('-al', "--alpha", type=float, 
                        default=0.1, help="Alpha")
    parser.add_argument('-the', "--theta", type=float, 
                        default=0.001, help="Theta")
    parser.add_argument('-emp', "--num_employedBees", type=int, 
                        default=0, help="Number of employed bees (Equal to k according to paper)")
    parser.add_argument('-onl', "--num_onlookers", type=int, 
                        default=0, help="Number of onlookers (Equal to k according to paper)")
    parser.add_argument('-iter', "--max_iteration", type=int, 
                        default=20000, help="Maximum iteration")
    parser.add_argument('-lim', "--limit", type=int, 
                        default=0, help="Limit (~50n according to paper)")
    
    return parser.parse_args()
    
def main(args):
    # limit = 50n, m >= 3, (n+m) >= 7
    n = args.num_customers
    m = args.num_vehicles
    k = args.num_solutions # k = 25
    c = args.capacity
    alpha = args.alpha # according to paper
    theta = args.theta # according to the paper
    
    if args.num_employedBees == 0:
        employedBees = k
    if args.num_onlookers == 0:
        onlookers = k

    #VRPProb = VRP(n, m, k)
    abc = ABC(n,m,k,c,alpha,theta,employedBees,onlookers, args.input_file)

    listOfFoodSources = abc.process(args.max_iteration, args.limit, args.input_file)
    
    for foodSource in listOfFoodSources:
        print(foodSource)

    chosenInd = 0
    maxVal = abc.calFitness(listOfFoodSources[chosenInd])
    for (i,foodsource) in enumerate(listOfFoodSources):
        val = abc.calFitness(foodsource)
        if val > maxVal:
            chosenInd = i
            maxVal = val

    abc.visualize(listOfFoodSources[chosenInd])
    '''VRPProb.readData('D:\\University\\Nam 4 HK 1\\Soft computing\\DoAn_CK\\code\\data\\Problem_8.txt')
    solList = VRPProb.initSols()

    for sol in solList:
        print('[+] normal: ', sol)
        print('[+] swapReverse: ', swapReverse_neighborOps(sol))'''

# Parse Args
args = ParseArguments()

if __name__ == "__main__":
    main(args)