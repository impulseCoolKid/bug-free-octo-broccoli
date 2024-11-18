import cond_color
import numpy 
import matplotlib.pyplot as plt
import part1 


#init
ALLCONDITIONS = 108
ALLTIME = 130
ALLNEURONS = 182

data = numpy.load("psths.npz")
X, times = data["X"], data["times"]


# Plot a histogram of the neurons’ maximum (across conditions and time) firing rates

dataBinsForAggregate = [] #numpy.zeros(ALLTIME)

for neuron in range(ALLNEURONS):
    for condition in range(ALLCONDITIONS):
       #dataBinsForAggregate[numpy.argmax(X[neuron][condition])] += 1
        dataBinsForAggregate.append(max(X[neuron][condition]))

#histogram of neuron maximums but im pretty sure this one has errors  
'''plt.plot( numpy.arange(-800, 500, 10)[1:ALLTIME-1], dataBinsForAggregate[1:ALLTIME-1])
plt.xlabel('ms')
plt.ylabel('neurons maximum firing ')
plt.title("histogram of the neurons maximum (across conditions and time)")
# Show the plot
plt.show()'''

#histogram of neuron maximums
"""plt.hist(dataBinsForAggregate, bins=600, edgecolor='blue')  # 'bins' defines the number of intervals

# Add labels and a title
plt.xlabel('firing rate')
plt.ylabel('Frequency of max firing rate')
plt.title('Histogram of max firing rates')

# Show the plot
plt.show()"""

#for Then, separately for each neuron, 3 normalize its PSTH according to: psth = (psth - b) / (a - b + 5)

PSTHALL = part1.normalData


activeNeuron = X[10]

def activateAll(activeNeuron):
    a = activeNeuron.max()#max([max(x) for x in activeNeuron])
    b = activeNeuron.min()#min([min(x) for x in activeNeuron])
    """allConditions = sum(activeNeuron)
    psth = (allConditions -b)/(a-b+5)"""
    psth = [(con -b)/(a-b+5) for con in activeNeuron]
    return psth

#normalizedNeuron
X = [activateAll(x) for x in X]


#TODO ammend this code so it works with code at the end then finish for the day


#cross condition mean
conditionSum = [sum(activeNeuron) for activeNeuron in X]

crossConditionMean = numpy.mean(conditionSum, axis=1)
XwithoutCCMean = [X[inter] - crossConditionMean[inter] for inter in range(182)]

#X = [X[index]/crossConditionMean[index] for index in range()]

X = numpy.array(XwithoutCCMean)[:,:,-65:-20]
#print(len(X[0])) #45 time bins



'''plt.plot(numpy.arange(-150, 300, 10),X[44])
plt.show()'''

#use PCA to obtain a dimensionality-reduced version of it by projecting 
# onto the first M = 12 principle components in the neuron activity space

#reshape array
X = X.reshape(182, 108*45)


C = X @ X.T / (108*45 - 1)
eigenvalues, eigenvectors = numpy.linalg.eig(C)

# Sort the eigenvalues and eigenvectors in descending order
# Get indices that would sort the eigenvalues in descending order
sorted_indices = numpy.argsort(numpy.real(eigenvalues))[::-1]

# Sort eigenvalues and eigenvectors using these indices
sorted_eigenvalues = numpy.real(eigenvalues)[sorted_indices]
V_m = numpy.real(eigenvectors)[:, sorted_indices]

Z = V_m[:,: 12].T @ X
#print(numpy.shape(Z))

'''plt.plot(Z[0],Z[1])
plt.show()'''

#cond_color.get_colors(Z[0],Z[1])