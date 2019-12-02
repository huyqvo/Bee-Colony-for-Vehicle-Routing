from initialization import VRP

n = int(input('Enter n:'))
m = int(input('Enter m:'))
k = int(input('Enter k:'))
VRPProb = VRP(n, m, k)

VRPProb.readData('D:\\University\\Nam 4 HK 1\\Soft computing\\DoAn_CK\\code\\data\\Problem_8.txt')
solList = VRPProb.initSols()

for sol in solList:
    print(sol)
