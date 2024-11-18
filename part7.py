import cond_color
import numpy as np
import matplotlib.pyplot as plt
import part5
import part2
import part6
import part1
import part4

#for diffrent neurons on an indipendent basis randomly select half the control conditions 
CONDITIONS = 108
NEURONS = 182
X = part2.XwithoutCCMean 
t0 = 65
T = 130

for neu in range(NEURONS):
    ConditionSelectionIndicies = np.random.choice(CONDITIONS, (CONDITIONS//2,), replace=False)
    
    for con in ConditionSelectionIndicies:
        x = X[neu][con]
        X[neu][con][t0:T] = 2 * x[t0] - x[t0:T]
#invert these randomly selected neuyrons using x[t0:T] = 2 * x[t0] - x[t0:T]
#rerun  code


def getZforinterval(START = -65,END  = -20, X = X):
    X = np.array(X)[:,:,-65:-20]
    X = X.reshape(182, 108*(END-START))
    C = X @ X.T / (108*(END-START))
    eigenvalues, eigenvectors = np.linalg.eig(C)
    sorted_indices = np.argsort(np.real(eigenvalues))[::-1]
    sorted_eigenvalues = np.real(eigenvalues)[sorted_indices]
    V_m = np.real(eigenvectors)[:, sorted_indices]
    Z = V_m[:,: 12].T @ X

    try:
        Z = Z.reshape(12, 108, END-START)
    except ValueError as e:
        print("Error while reshaping:", e)

    return Z, V_m[:,: 12]

#part2 

Z, V_m  = getZforinterval()
A = part4.findA(Z=Z)
eigenvalues, eigenvectors = np.linalg.eig(A)
P = part5.getProjectionMatrix()
P_fr = np.tensordot(P.T,Z,axes=([1], [0]))
PxV_mT= np.tensordot(P.T,V_m.T,axes=([1], [0]))
X_start = np.array(X)[:,:,0:65]
XProjection_begining = np.tensordot(PxV_mT,X_start,axes=([1], [0]))


part5.printGraphs(P_fr=P_fr[:,:,:-10])
plt.show() #unccoment to show graph
