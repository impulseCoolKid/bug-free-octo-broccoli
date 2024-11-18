import cond_color
import numpy as np
import matplotlib.pyplot as plt
import part5
import part2
import part1 

BEGINEINGTO = 65

def getZforinterval(START = 0,END  = 65):
    X = np.array(part2.XwithoutCCMean)[:,:,START:END]
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

    return Z, V_m

Z_start, V_m = getZforinterval(END=65)


'''P_0 = part5.getProjectionMatrix(0)
P_fr = np.tensordot(P_0.T,Z_start,axes=([1], [0]))# dont use this PFR
part5.printGraphs(P_fr,BEGINEINGTO-1)'''


P = part5.P #i used p instead of p_fr since i wanted a projection non a manifold
P_fr = part5.P_fr
V_m = part2.V_m[:,: 12]

#option 1 plot onto 2 dominant eigen vectors giviing us

PxV_mT= np.tensordot(P.T,V_m.T,axes=([1], [0]))


#X_start 
X_start = np.array(part2.XwithoutCCMean)[:,:,0:65]
print(np.shape(X_start))
XProjection_begining = np.tensordot(PxV_mT,X_start,axes=([1], [0]))


'''part5.printGraphs(XProjection_begining,64,True,1, title = "Begining projections")
part5.printGraphs(alpha = 0.2,title = "projections on manifold with starting conditions")
plt.show() #unccoment to show graph


'''