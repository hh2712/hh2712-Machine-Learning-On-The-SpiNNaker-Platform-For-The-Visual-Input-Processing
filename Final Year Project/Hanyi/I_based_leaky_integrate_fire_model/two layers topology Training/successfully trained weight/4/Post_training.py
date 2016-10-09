__author__ = 'hanyi'
import spynnaker.pyNN as sim
import pylab
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


cell_params_lif = {
                   'cm': 1.9,
                   'i_offset': 0.0,
                   'tau_m': 110,
                   'tau_refrac': 150.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 10.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -60.0 
}

NetworkInfo = scipy.io.loadmat('trained_weight.mat')
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
postpop_size = 20
animation_time = 200
episode = 200
order = np.array(range(36))
test_order = np.array([0,1,2,3])
simtime = len(order)*animation_time+8*animation_time

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
        pylab.ylim((0, postpop_size+2))
        lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],".")
        pylab.axvspan(1800,2200 , facecolor='b', alpha=0.5)
        pylab.axvspan(4000,4400 , facecolor='b', alpha=0.5)
        pylab.axvspan(6200,6600 , facecolor='b', alpha=0.5)
        pylab.axvspan(8400,8800 , facecolor='b', alpha=0.5)
        pylab.setp(lines,markersize=10,color='r')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
             item.set_fontsize(20)
        pylab.show()

    else:
        print "No spikes received"
'''(x_r, y_r, p_r, ts_r) = get_data('downsampled_left_to_right_2.mat')
(x_l, y_l, p_l, ts_l) = get_data('downsampled_right_to_left_2.mat')
(x_d, y_d, p_d, ts_d) = get_data('downsampled_top_to_bottom_2.mat')
(x_u, y_u, p_u, ts_u) = get_data('downsampled_bottom_to_top_2.mat')'''
(x_r1, y_r1, p_r1, ts_r1) = get_data('l_to_r1.mat')
(x_r2, y_r2, p_r2, ts_r2) = get_data('l_to_r2.mat')
(x_r3, y_r3, p_r3, ts_r3) = get_data('l_to_r3.mat')
(x_r4, y_r4, p_r4, ts_r4) = get_data('l_to_r4.mat')
(x_r5, y_r5, p_r5, ts_r5) = get_data('l_to_r5.mat')
(x_r6, y_r6, p_r6, ts_r6) = get_data('l_to_r6.mat')
(x_r7, y_r7, p_r7, ts_r7) = get_data('l_to_r7.mat')
(x_r8, y_r8, p_r8, ts_r8) = get_data('l_to_r8.mat')
(x_r9, y_r9, p_r9, ts_r9) = get_data('l_to_r9.mat')

(x_l1, y_l1, p_l1, ts_l1) = get_data('r_to_l1.mat')
(x_l2, y_l2, p_l2, ts_l2) = get_data('r_to_l2.mat')
(x_l3, y_l3, p_l3, ts_l3) = get_data('r_to_l3.mat')
(x_l4, y_l4, p_l4, ts_l4) = get_data('r_to_l4.mat')
(x_l5, y_l5, p_l5, ts_l5) = get_data('r_to_l5.mat')
(x_l6, y_l6, p_l6, ts_l6) = get_data('r_to_l6.mat')
(x_l7, y_l7, p_l7, ts_l7) = get_data('r_to_l7.mat')
(x_l8, y_l8, p_l8, ts_l8) = get_data('r_to_l8.mat')
(x_l9, y_l9, p_l9, ts_l9) = get_data('r_to_l9.mat')

(x_b1, y_b1, p_b1, ts_b1) = get_data('t_to_b1.mat')
(x_b2, y_b2, p_b2, ts_b2) = get_data('t_to_b2.mat')
(x_b3, y_b3, p_b3, ts_b3) = get_data('t_to_b3.mat')
(x_b4, y_b4, p_b4, ts_b4) = get_data('t_to_b4.mat')
(x_b5, y_b5, p_b5, ts_b5) = get_data('t_to_b5.mat')
(x_b6, y_b6, p_b6, ts_b6) = get_data('t_to_b6.mat')
(x_b7, y_b7, p_b7, ts_b7) = get_data('t_to_b7.mat')
(x_b8, y_b8, p_b8, ts_b8) = get_data('t_to_b8.mat')
(x_b9, y_b9, p_b9, ts_b9) = get_data('t_to_b9.mat')

(x_t1, y_t1, p_t1, ts_t1) = get_data('b_to_t1.mat')
(x_t2, y_t2, p_t2, ts_t2) = get_data('b_to_t2.mat')
(x_t3, y_t3, p_t3, ts_t3) = get_data('b_to_t3.mat')
(x_t4, y_t4, p_t4, ts_t4) = get_data('b_to_t4.mat')
(x_t5, y_t5, p_t5, ts_t5) = get_data('b_to_t5.mat')
(x_t6, y_t6, p_t6, ts_t6) = get_data('b_to_t6.mat')
(x_t7, y_t7, p_t7, ts_t7) = get_data('b_to_t7.mat')
(x_t8, y_t8, p_t8, ts_t8) = get_data('b_to_t8.mat')
(x_t9, y_t9, p_t9, ts_t9) = get_data('b_to_t9.mat')

argw =((x_r1, y_r1, p_r1, ts_r1),(x_r2, y_r2, p_r2, ts_r2),
       (x_r3, y_r3, p_r3, ts_r3),(x_r4, y_r4, p_r4, ts_r4),
       (x_r5, y_r5, p_r5, ts_r5),(x_r6, y_r6, p_r6, ts_r6),
       (x_r7, y_r7, p_r7, ts_r7),(x_r8, y_r8, p_r8, ts_r8),
       (x_r9, y_r9, p_r9, ts_r9),(x_l1, y_l1, p_l1, ts_l1),
       (x_l2, y_l2, p_l2, ts_l2),(x_l3, y_l3, p_l3, ts_l3),
       (x_l4, y_l4, p_l4, ts_l4),(x_l5, y_l5, p_l5, ts_l5),
       (x_l6, y_l6, p_l6, ts_l6),(x_l7, y_l7, p_l7, ts_l7),
       (x_l8, y_l8, p_l8, ts_l8),(x_l9, y_l9, p_l9, ts_l9),
       (x_b1, y_b1, p_b1, ts_b1),(x_b2, y_b2, p_b2, ts_b2),
       (x_b3, y_b3, p_b3, ts_b3),(x_b4, y_b4, p_b4, ts_b4),
       (x_b5, y_b5, p_b5, ts_b5),(x_b6, y_b6, p_b6, ts_b6),
       (x_b7, y_b7, p_b7, ts_b7),(x_b8, y_b8, p_b8, ts_b8),
       (x_b9, y_b9, p_b9, ts_b9),(x_t1, y_t1, p_t1, ts_t1),
       (x_t2, y_t2, p_t2, ts_t2),(x_t3, y_t3, p_t3, ts_t3),
       (x_t4, y_t4, p_t4, ts_t4),(x_t5, y_t5, p_t5, ts_t5),
       (x_t6, y_t6, p_t6, ts_t6),(x_t7, y_t7, p_t7, ts_t7),
       (x_t8, y_t8, p_t8, ts_t8),(x_t9, y_t9, p_t9, ts_t9))

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
    weights = 3,delays=1,allow_self_connections=False), target='inhibitory')
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
for jj in range(0,44):
    rate, table =  GetFiringPattern(post_spikes,200*jj,200*(jj+1))
    print table,jj
    sec_layer_firing_rate.append(rate)
    sec_layer_firing_table.append(table)

scipy.io.savemat('trained_firing_info.mat',{'firing_rate':sec_layer_firing_rate,
                                       'firing_table':sec_layer_firing_table})    

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
sup_sub_checker(sec_layer_firing_table)
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


