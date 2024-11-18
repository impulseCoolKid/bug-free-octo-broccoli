import cond_color
import numpy 
import matplotlib.pyplot as plt

DATAPATH = "psths.npz" 
data = None
X, times = None,None

data = numpy.load(DATAPATH)
'''Z_test = data['Z_test']
A_test = data['A_test']
print(numpy.shape(Z_test))
X = Z_test'''
X, times = data["X"], data["times"]




#the variable 
CONDITION = 100 #out of 108
ALLCONDITIONS = 108
ALLTIME = 130
ALLNEURONS = 182

# this creates nice sub plots
# Create subplots (3 rows, 1 column)
fig, axes = plt.subplots(8, 1, figsize=(15, 6))  # Adjust 'figsize' as needed
neuronNumbers =''
for item in range(8):
    axes[item].plot( numpy.arange(-800, 500, 10), X[item*22+3][CONDITION])#, bins=130, edgecolor='black') # x[neuron number, condition]
    neuronNumbers += str(item*22+3) + ","

plt.xlabel('ms')
fig.suptitle('PSTH Condition:{0}, with neurons {1}'.format(CONDITION,neuronNumbers), fontsize=16)
# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()

# Agregate data plots
#sum all the neuron data at a given time 

AgregateData = numpy.zeros(ALLTIME)
for neuron in range(ALLNEURONS):
    for condition in range(ALLCONDITIONS):
        for timeStep in range(ALLTIME):
            AgregateData[timeStep] += X[neuron][condition][timeStep]

normalData = AgregateData/ (ALLNEURONS*ALLCONDITIONS)

"""plt.plot( numpy.arange(-800, 500, 10), normalData)
plt.axhline(y=2.625, color='b', linestyle='--', label='Horizontal Line 1 (y=4)')
plt.xlabel('ms')
plt.ylabel('adverage neuron spikes')
plt.title("Agregate data")
# Show the plot
#plt.show()"""