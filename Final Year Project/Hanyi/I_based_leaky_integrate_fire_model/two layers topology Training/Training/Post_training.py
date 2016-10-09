__author__ = 'hanyi'
import spynnaker.pyNN as sim
import pylab
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


cell_params_lif = {
                   'cm': 12,
                   'i_offset': 0.0,
                   'tau_m': 110,
                   'tau_refrac': 40.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 10.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -61.0 
}

sim.setup(timestep=1, min_delay=1, max_delay=144)
delay = 1
prepop_size = 256
postpop_size = 4
animation_time = 200
episode = 200
order = np.array(range(36))
test_order = np.array([0,1,2,3])
simtime = len(order)*animation_time+8*animation_time

NetworkInfo = scipy.io.loadmat('trained_weight_65.mat')
weights_import = NetworkInfo['trained_weight']

def convert_weights_to_list(matrix, delay):
    def build_list(indices):
        # Extract weights from matrix using indices
        weights = matrix[indices]
        # Build np array of delays
        delays = np.repeat(delay, len(weights))
        # Zip x-y coordinates of non-zero weights with weights and delays
        return zip(indices[0], indices[1], weights, delays)

    # Get indices of non-nan i.e. connected weights
    connected_indices = np.where(~np.isnan(matrix))
    # Return connection lists
    return build_list(connected_indices)

def Plot_WeightDistribution(weight,bin_num,title):
    hist,bins = np.histogram(weight,bins=bin_num)
    center = (bins[:-1]+bins[1:])/2
    width = (bins[1]-bins[0])*0.7
    plt.bar(center,hist,align='center',width =width)
    plt.xlabel('Weight')
    plt.title(title)
    plt.show()
#Plot_WeightDistribution(weights_import,200,'trained weight')


def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0]) #from ns to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts


def BuildTrainingSpike(order,ONOFF):
    complete_Time = []
    for nid in range(0,prepop_size):
        SpikeTimes = []
        for tid in range(0,len(order)):
                dead_zone_cnt = int(tid/9)
                #print dead_zone_cnt
                temp=[]
                loc = order[tid]
                j = np.repeat(nid,len(argw[loc][1]))
                p = np.repeat(ONOFF,len(argw[loc][1]))
                temp = 200*(tid+2*dead_zone_cnt)+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]
                if temp.size>0:
                   SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
           complete_Time.append(SpikeTimes.tolist())
        else:
            complete_Time.append([])
    return complete_Time

        
def plot_spikes(spikes, title):    
    if spikes is not None:
        pylab.figure()
        ax = plt.subplot(111, xlabel='Time/ms', ylabel='Neruons #', title=title)
        pylab.xlim((0, simtime))
        pylab.ylim((-0.5, postpop_size-0.5))
        lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],".")
        pylab.axvline(2000,linewidth = 4, color = 'c',alpha = 0.75,linestyle = 'dashed')
        pylab.axvline(4200,0,1,linewidth = 4, color = 'c',alpha = 0.75,linestyle = 'dashed')
        pylab.axvline(6400,0,1,linewidth = 4, color = 'c',alpha = 0.75,linestyle = 'dashed')
        '''pylab.axvspan(1800,2200 , facecolor='b', alpha=0.5)
        pylab.axvspan(4000,4400 , facecolor='b', alpha=0.5)
        pylab.axvspan(6200,6600 , facecolor='b', alpha=0.5)
        pylab.axvspan(8400,8800 , facecolor='b', alpha=0.5)'''
        pylab.setp(lines,markersize=10,color='r')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
             item.set_fontsize(20)
        pylab.show()

    else:
        print "No spikes received"

argw = []

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0]) #from us to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts

def data_set(prefix):
    postfix = '.mat'
    for i in range(1,10):
            argw.append(get_data(prefix+str(i)+postfix))
    
data_set('l_to_r')
data_set('r_to_l')
data_set('t_to_b')
data_set('b_to_t')

#Let us only use the ON events
TrianSpikeON = BuildTrainingSpike(order,1)
#print TrianSpikeON
spikeArrayOn = {'spike_times': TrianSpikeON}
ON_pop = sim.Population(prepop_size, sim.SpikeSourceArray, spikeArrayOn,
                        label='inputSpikes_On')
post_pop= sim.Population(postpop_size,sim.IF_curr_exp, cell_params_lif,
                         label='post_1')


connectionsOn = sim.Projection(ON_pop, post_pop, sim.FromListConnector(
    convert_weights_to_list(weights_import, delay)))   
#inhibitory between the neurons
connection_I  = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(
    weights = 15,delays=1,allow_self_connections=False), target='inhibitory')
post_pop.record()

sim.run(simtime)

# == Get the Simulated Data =================================================
post_spikes = post_pop.getSpikes(compatible_output=True)
sim.end()


def plot_spike_histogram(spikes, bin_num,title):
    hist,bins = np.histogram([i[0] for i in spikes],bins=bin_num)
    center = (bins[:-1]+bins[1:])/2
    width = (bins[1]-bins[0])*0.7
    ax = pylab.subplot(111,xlabel='Neuron ID',ylabel = 'spiking times',title =title)
    plt.bar(center,hist,align='center',width =width)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.show()
    
    
plot_spike_histogram(post_spikes,postpop_size,"Post-Synaptic Neuron Spiking Rate ")    
plot_spikes(post_spikes, "Spike Pattern of Post-Synaptic Population")


