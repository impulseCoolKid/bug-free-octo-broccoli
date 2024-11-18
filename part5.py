import cond_color
import numpy as np
import matplotlib.pyplot as plt
import part4

#copy data over
Z = part4.Z
eigenvalues, eigenvectors = np.linalg.eig(part4.A)

#which scale eigen value
EIGEN_INDEX = 2

def getProjectionMatrix(EIGEN_INDEX = EIGEN_INDEX ):
    #sort eigen stuff
    sorted_indices = np.argsort(np.imag(eigenvalues))#[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenVectors = eigenvectors[:, sorted_indices]

    #form P
    normalising_value  = 1#np.imag(sorted_eigenvalues[EIGEN_INDEX ])
    col1 = np.array(np.real(eigenvectors[EIGEN_INDEX ])/ normalising_value)
    col2 = np.array(np.imag(eigenvectors[EIGEN_INDEX ])/normalising_value)

    P = np.array([col1,col2]).T
    P = P/P.max() #TODO is this normalisation fine???
    
    return P

P = getProjectionMatrix()
#project P on Z
P_fr = np.tensordot(P.T,Z,axes=([1], [0]))
#print(np.shape(P_fr))
#plotting

#marker sizes
BEGIN_SIZE =200
END_SIZE =100

def printGraphs(P_fr=P_fr[:,:,:-10],end = 34,alt_colors = False,alpha=1,**title):
    for i in range(len(P_fr[0][1])):
        colours = cond_color.get_colors(P_fr[0][i],P_fr[1][i],alt_colors=alt_colors)
        plt.plot(P_fr[0][i], P_fr[1][i], color=colours[i],alpha = alpha)

        cond_color.plot_start(P_fr[0][i][0],P_fr[1][i][0],colours[i],markersize=BEGIN_SIZE)
        cond_color.plot_end(P_fr[0][i][end],P_fr[1][i][end],colours[i],markersize=END_SIZE)

    if len(title) == 0:
        plt.title('Rotation Dynamics on 2D manifold based on eigen value at index:{0}'.format(EIGEN_INDEX))
    else:
        plt.title(title)
    plt.xlabel('Real projection')
    plt.ylabel('Imaginary projection')

    plt.show() #unccoment to show graph

printGraphs()#TODO uncomment to get graphs