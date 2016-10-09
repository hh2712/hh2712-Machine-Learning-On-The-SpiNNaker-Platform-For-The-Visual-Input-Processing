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

sim.setup(timestep=1, min_delay=1, max_delay=144)
synapses_to_spike = 1
delay = 1
prepop_size = 256
postpop_size = 4
animation_time = 200
episode = 200
order = np.array(range(44))
simtime = len(order)*2*animation_time

def concatenate_time(time,iter):
    temp_time = []
    spike_time= []
    for kk in range(0,iter):
        spike_time = np.concatenate((temp_time,time+kk*animation_time*4))
        temp_time = spike_time
    return temp_time
#Train_time =  concatenate_time(firing_time,len(order)/4)
#NeuronID = np.tile(NetworkInfo['NeuronID'][0],len(order)/4)

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0]) #from ns to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts

def ReadSpikeTime(NeuronID,x,y,ts,p,ONOFF):
    timeTuple=[]
    for idx in range(0,len(x)):
        if NeuronID == (x[idx]+y[idx]*16) and p[idx]==ONOFF:
           timeTuple.append(ts[idx])
    return timeTuple

def BuildSpike(x,y,ts,p,ONOFF):
    SpikeTimes = []
    for i in range(0,prepop_size):
        SpikeTimes.append(ReadSpikeTime(i,x,y,ts,p,ONOFF))
    return SpikeTimes

def BuildTrainingSpike(order,ONOFF):
    complete_Time = []
    for nid in range(0,prepop_size):
        SpikeTimes = []
        for tid in range(0,len(order)):
                #dead_zone_cnt = int(tid/9)
                #print dead_zone_cnt
                temp=[]
                loc = order[tid]
                j = np.repeat(nid,len(argw[loc][1]))
                p = np.repeat(ONOFF,len(argw[loc][1]))
                '''temp = 200*(tid+2*dead_zone_cnt)+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]'''
                temp = 200*(2*tid+1)+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]
                if temp.size>0:
                   SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
            complete_Time.append(SpikeTimes.tolist())
        else:
            complete_Time.append([])
    return complete_Time

def compare_spikes(spikes, title):
    if spikes is not None:
        pylab.figure()
        ax = plt.subplot(111, xlabel='Time/ms', ylabel='Neruons #', title=title)
        pylab.xlim((0, simtime))
        pylab.ylim((0, postpop_size+2))
        line1 = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],
                           'r|',label='post-train spikes')
        line2 = pylab.plot(Train_time,NeuronID,'b|',label='trained spikes')
        pylab.setp(line1,markersize=10,linewidth=25)
        pylab.setp(line2,markersize=10,linewidth=25)
        pylab.legend()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
             item.set_fontsize(20)
        pylab.show()
    else:
        print "No spikes received"
        
def plot_spikes(spikes, title):    
    if spikes is not None:
        pylab.figure()
        ax = plt.subplot(111, xlabel='Time/ms', ylabel='Neruons #', title=title)
        pylab.xlim((0, simtime))
        pylab.ylim((-0.5, postpop_size-0.5))
        lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],".")
        pylab.setp(lines,markersize=10,color='r')
        for i in range(3):
            #pylab.axvspan(i*600,(i+1)*600, facecolor='b', alpha=0.5)
            pylab.axvline((i+1)*4400,0,1,linewidth = 4, color = 'c',alpha = 0.75,linestyle = 'dashed')
        '''pylab.axvspan(1800,2200 , facecolor='b', alpha=0.5)
        pylab.axvspan(4000,4400 , facecolor='b', alpha=0.5)
        pylab.axvspan(6200,6600 , facecolor='b', alpha=0.5)
        pylab.axvspan(8400,8800 , facecolor='b', alpha=0.5)
        '''
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
             item.set_fontsize(20)
        '''major_ticksx = np.arange(0, 101, 20)                                              
        minor_ticksx = np.arange(0, 101, 5)
        ax.set_xticks(major_ticksx)                                                       
        ax.set_xticks(minor_ticksx, minor=True)'''
        pylab.show()

    else:
        print "No spikes received"
        
argw = []
def data_set(prefix):
    postfix = '.mat'
    mid_str = '_'
    num_dataset = 1
    for i in range(50,160,10):
        '''if (i==0):
            num_dataset = 5
        else:
            num_dataset = 3
            mid_str = '0d_new'''
        for j in range(1,num_dataset+1):
            argw.append(get_data(prefix+str(i)+postfix))
    
data_set('new_l2r_')
data_set('new_r2l_')
data_set('new_t2b_')
data_set('new_b2t_')
print len(argw)

def raster_plot():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0,len(argw)):
        pylab.plot(argw[ii][3]+200*ii,argw[ii][0]+argw[ii][1]*16,".")
    pylab.title('raster plot of Virtual Retina Neuron Population')
    pylab.show()

raster_plot()
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

def GetFiringPattern(spike,low,high):
    spikeT = np.transpose(spike)
    time_stamp = spikeT[1]
    target_index = ((time_stamp-low)>=0) & ((time_stamp-high)<0)
    firingTable = np.unique(spikeT[0][target_index])
    firingRate = len(np.unique(spikeT[0][target_index]))
    return firingRate,firingTable

sec_layer_firing_rate = []
sec_layer_firing_table= []
'''for jj in range(0,44):
    rate, table =  GetFiringPattern(post_spikes,200*jj,200*(jj+1))
    print table,jj
    sec_layer_firing_rate.append(rate)
    sec_layer_firing_table.append(table)'''

#scipy.io.savemat('trained_firing_info.mat',{'firing_rate':sec_layer_firing_rate,
#                                       'firing_table':sec_layer_firing_table})    

def check_uniq(p_subset,p_superset):
    flag = 0;#when flag is 1, it means two sets are not subset and superset
    j = 0
    i = 0
    while(i != len(p_subset)):
        if(p_subset[i]==p_superset[j]):
            i += 1
        else:
            j += 1
            if(j==len(p_superset)):
                flag = 1
                break
        
    if(flag == 0 ):
        print "there exists subset"
    return flag
            
def sup_sub_checker(firing_table):
    #find the len order
    flag = 1
    for i in range (4):
        for j in range (4):
            if(len(firing_table[i])<=len(firing_table[j]) and i !=j ):
                flag = flag & check_uniq(firing_table[i],firing_table[j])
    if(flag == 1):
        print "all clear"
        
#sup_sub_checker(sec_layer_firing_table)
def plot_spike_histogram(spikes, bin_num,title):
    print [i[0] for i in spikes]
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
