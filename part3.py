import part2
import cond_color
import numpy as np
import matplotlib.pyplot as plt

#Bring in variables
Z = part2.Z
#V_m = part2.V_m

#reshape Z
try:
    Z_reshape = Z.reshape(12, 108, 45)
    #print("Original shape:", Z.shape)
    #print("Reshaped shape:", Z_reshape.shape)
except ValueError as e:
    print("Error while reshaping:", e)

Z =Z_reshape
#print(np.shape(Z))

#marker sizes
BEGIN_SIZE =200
END_SIZE =100

'''for i in range(len(Z[0][1])):
    colours = cond_color.get_colors(Z[0][i],Z[1][i])
    plt.plot(Z[0][i], Z[1][i], color=colours[i])

    cond_color.plot_start(Z[0][i][0],Z[1][i][0],colours[i],markersize=BEGIN_SIZE)
    cond_color.plot_end(Z[0][i][44],Z[1][i][44],colours[i],markersize=END_SIZE)

plt.title('Neuron Activity Trajectories Mapped onto the 1st and 2nd Principal Components')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
'''
#plt.show() #unccoment to show graph