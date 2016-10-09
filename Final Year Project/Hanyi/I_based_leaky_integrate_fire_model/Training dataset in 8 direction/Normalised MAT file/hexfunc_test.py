import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
NetworkInfo = scipy.io.loadmat('trained_weight.mat')
weights_import = NetworkInfo['trained_weight']

prepop_size = 256
postpop_size = 8

def Plot_WeightDistribution_in_2d(weight,title):
    #print weight
    xrange = np.arange(1,17)
    yrange = np.arange(1,17)
    xx, yy = np.meshgrid(xrange, yrange)
    xx = np.reshape(xx, (1,np.product(xx.shape)))[0]
    yy = np.reshape(yy, (1,np.product(yy.shape)))[0]
    #plt.hist2d(xx, yy, C = weight,bins='log', cmap=plt.cm.YlOrRd_r)
    plt.hist2d(xx, yy, weights = weight, bins=16)
    plt.axis([1, 16, 1, 16])
    plt.title(title)
    cb = plt.colorbar()
    
    cb.set_label('Weight')
    plt.show()
    
for i in range(postpop_size):
    Plot_WeightDistribution_in_2d([x[i] for x in weights_import],"neuron"+str(i)+" synapse weights 2D plot")
#Plot_WeightDistribution_in_2d([x[0] for x in weights_import],"neuron 0 synapse weights 2D plot")