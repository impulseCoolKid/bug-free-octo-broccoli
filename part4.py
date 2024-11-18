import part3
import cond_color
import numpy as np
import matplotlib.pyplot as plt

#TESTING

DATAPATH = "test.npz" #psths.npz" 
data = np.load(DATAPATH)
Z_test = data['Z_test']
A_test = data['A_test']


M = 12 # m = 12, K =66
K = 66
T=45
Z = part3.Z#[:M,1]
CONDITIONS = 108


# part b 
def makeH(dim=M,K= K):
    #K = (dim*dim/2)-dim/2
    H = np.zeros((K, dim,dim), dtype=int)
   
    a = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            H[a, i, j] = 1
            H[a, j, i] = -1
            a += 1
    return H


def findA(Z=Z,T=T,K=K,M=M):
    H = makeH(M,K)
    sumationInverse = np.zeros((K, K))
    sumationReg = np.zeros((K, K))

    for counter in range(T-1):
        for cond in range(CONDITIONS):
            Z_t = np.array(Z[:,cond,counter]).reshape(-1, 1)
            W =  np.tensordot(H, Z_t.T[0], axes=1)
            Z_tplusone = np.array(Z[:,cond,counter +1]).reshape(-1, 1)
            delZ = Z_tplusone-Z_t 
            multi = delZ.T @ W.T
            multiInv = W @ W.T
            sumationInverse += multiInv
            sumationReg += multi

    sumationInverse = np.linalg.inv(sumationInverse)
    beta = sum(sumationReg @ sumationInverse)

    A = np.tensordot(beta, H, axes=1)
    return A#/66 # dodgy number but seems to work


A = findA(np.array(Z))
#TESTING
'''
DATAPATH = "test.npz" #psths.npz" 

data = np.load(DATAPATH)
Z_test = data['Z_test']
A_test = data['A_test']
print(np.shape(Z_test))

A = findA(np.array(Z_test))

A=A/66



matCHeck = A-A_test
SD = 0
total = 0
for i in matCHeck:
    for j in i:
        total += 1
        SD += j*j
print(SD/ total)


# Sum the elements of the difference matrix
difference_sum = (A).max()-A_test.max()
print(difference_sum)


plt.figure()
plt.imshow(A_test, cmap='viridis')  # 'viridis' is just one colormap option
plt.colorbar()  # Display the color scale bar
plt.title('matrix: A')
plt.xlabel('row')
plt.ylabel('column') 
plt.figure()
plt.imshow(A, cmap='viridis')
plt.colorbar()  # Display the color scale bar
plt.title('matrix: A via Z_test data')
plt.xlabel('row')
plt.ylabel('column') 
plt.show()
'''


