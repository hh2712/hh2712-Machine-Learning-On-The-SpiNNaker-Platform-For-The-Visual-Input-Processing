__author__ = "hanyi"
import I_based_model as I_sim
import numpy as np
from types import FunctionType
import random
import scipy.io
import pylab
import matplotlib.pyplot as plt

inhibitory_mode = False
test_mode = False
Normal_training_mode = True

if(Normal_training_mode):
    pre_pop_size = 256
    post_pop_size = 1
    test_STDP = False
    STDP_mode = True
    inhibitory_spike_mode = False
    allsameweight = False
    self_import = False
else:
    test_STDP = True
    if(inhibitory_mode):
        pre_pop_size = 4
        post_pop_size = 1
        STDP_mode =False
        inhibitory_spike_mode = True
        allsameweight = True
        self_import = False
    if(test_mode): 
        pre_pop_size = 256
        post_pop_size = 1
        STDP_mode =True
        inhibitory_spike_mode = False
        allsameweight = False
        self_import = True
        
#E_syn_weight = np.random.normal(0.2, 0.003,pre_pop_size*post_pop_size).tolist()
E_syn_weight = [0.2,0.1,0.2,0]
I_syn_weight = 15

setup_cond = {
                    'timestep': 1,
                    'min_delay':1,
                    'max_delay':144                    
} 

stdp_param = {
                    'tau_plus': 40.0,
                    'tau_minus': 60.0,
                    'w_min': 0,
                    'w_max': 1,
                    'A_plus': 0.05,
                    'A_minus': 0.05
                    
}

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

sim1 = I_sim.IF_curr_exp_s(pre_pop_size = pre_pop_size, post_pop_size = post_pop_size, 
                            E_syn_weight = E_syn_weight, I_syn_weight = I_syn_weight,
                            cell_params_lif = cell_params_lif, setup_cond = setup_cond,
                            stdp_param = stdp_param,STDP_mode = STDP_mode,inhibitory_spike_mode = inhibitory_spike_mode,allsameweight=allsameweight)

#--------------------------------------------------#
# the following input spike are recorded from DVS
#--------------------------------------------------#   
last_episodes = np.repeat(np.array([0,9,18,27]),1)
order = []
animation_time = 200

def create_shuffle_seq(num_copy):
    order = np.array([1])
    order_temp = [] 
    for i in range (4):
        for j in range (4):
            if (i!=j):
                for k in range (4):
                    if(i!=k and j!=k):
                        for z in range (4):
                            if(z!=i and z!=j and z!=k):
                                order_temp.append(9*i+random.randrange(0, 9))
                                order_temp.append(9*j+random.randrange(0, 9))
                                order_temp.append(9*k+random.randrange(0, 9))
                                order_temp.append(9*z+random.randrange(0, 9))
    np_order_temp =  np.array(order_temp)
    #1 in the np.array to initialise the data type to integer
    for i in range (num_copy):
        order = np.concatenate((order, np_order_temp))
    return order
    
def create_order(num_copy):
    order = np.array([1])
    order_temp = []
    '''for i in range(8):
        for j in range(num_copy):
            order_temp.append(random.randrange(i*9, (i+1)*9))'''
    for j in range(num_copy):
            order_temp.append(random.randrange(0, 9))
    order = np.concatenate((order,order_temp))
    return order

order = create_order(65)
#order = create_order(4)
    
#order = create_shuffle_seq(12)
#order1 = range(27,36)
'''for i in range(50):
    order = np.concatenate((order,order1))'''

print "---------------------------------------------"
#order = np.concatenate((order,last_episodes)) 
print order

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0]) #from us to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts


argw = []
def data_set():
    prefix = 'new'
    postfix = '.mat'
    mid_str = 'd_'
    num_dataset = 9
    for i in range(0,360,45):
        for j in range(1,num_dataset+1):
            argw.append(get_data(prefix+str(i)+mid_str+str(j)+postfix))
            
data_set()

'''def raster_plot_4_dir():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0,8):
        pylab.plot(argw[9*ii][3]+200*(2*ii+1),argw[9*ii][0]+argw[9*ii][1]*16,".")
    pylab.title('Raster Plot of Virtual Retina Neuron Population in 8 Direction')
    pylab.xlim((0, 3600))
    pylab.ylim((0, 270))
    pylab.show()'''
    
#raster_plot_4_dir()
    
def raster_plot():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0,len(argw)):
        pylab.plot(argw[ii][3]+200*(ii+ii/9),argw[ii][0]+argw[ii][1]*16,".")
    for i in range(1,9):
        pylab.axvline((2000*i-100),0,1,linewidth = 4, color = 'r',alpha = 0.75,linestyle = 'dashed')
    
    pylab.title('Raster Plot of 72 Neuron Population Training Sets')
    pylab.ylim((0, 270))
    pylab.show()

raster_plot()

'''NetworkInfo = scipy.io.loadmat('trained_weight.mat')
weights_import = NetworkInfo['trained_weight']'''
delay = 1

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
    

    
def BuildTrainingSpike(order,ONOFF):
    complete_Time = []
    for nid in range(0,pre_pop_size):
        SpikeTimes = []
        for tid in range(0,len(order)):
                #print dead_zone_cnt
                temp=[]
                loc = order[tid]
                j = np.repeat(nid,len(argw[loc][1]))
                p = np.repeat(ONOFF,len(argw[loc][1]))
                temp = 200*(2*tid+1)+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]
                if temp.size>0:
                   SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
           complete_Time.append(SpikeTimes.tolist())
        else:
            complete_Time.append([])
    return complete_Time

def BuildTrainingSpike_with_noise(order,ONOFF,noise_spikes):
    noise_nid = []
    for i in range(0,len(order)):
        noisetemp = np.random.randint(0,256,noise_spikes)
        noisetemp.sort()
        noisetemp = noisetemp.tolist()
        noise_nid.append(noisetemp)
    #print len(noise_nid)
    complete_Time = []
    for nid in range(0,pre_pop_size):
        SpikeTimes = []
        for tid in range(0,len(order)):
                #print dead_zone_cnt
                temp=[]
                loc = order[tid]
                j = np.repeat(nid,len(argw[loc][1]))
                p = np.repeat(ONOFF,len(argw[loc][1]))
                temp = 200*(2*tid+1)+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]
                if(nid in noise_nid[tid]):
                    t_noise = 200*(2*tid+1) + np.random.uniform(0,200,1)
                    temp = np.concatenate((temp,t_noise))
                    temp.sort()
                if temp.size>0:
                   SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
           complete_Time.append(SpikeTimes.tolist())
        else:
           complete_Time.append([])
    return complete_Time
    
#in_spike = BuildTrainingSpike(order,1)
#--------------------------------------------------# 

if(test_STDP):
    sim_time = 4000
    if(inhibitory_mode):
        in_spike = [[10],[],[],[]]
    else:
        #in_spike = [[0,10],[5],[20],[13]]
        #in_spike = [[0,10,100,110],[5,105],[20,120],[13,113]]
        #in_spike = [[0,10,100,110,200,210],[5,105,205],[20,120,220],[13,113,213]]
        #order = [1,2,3,4,5]
        in_spike = BuildTrainingSpike_with_noise(order,1,11)
        sim_time = (2*len(order)+1)*animation_time
else:
    in_spike = BuildTrainingSpike_with_noise(order,1,0)
    sim_time = (2*len(order)+1)*animation_time
    
sim1.input_spike(in_spike)
list = [(0, 0, 0.2, 1), (1, 0, 0.1, 1), (2, 0, 0.25, 1), (3, 0, 0.3, 1)]
#list = convert_weights_to_list(weights_import, delay)
#sim1.connection_list_converter_normal(self_import = self_import,conn_list = list, mean=0.06, var=0.001,delay = 1)
sim1.connection_list_converter_uniform(self_import = self_import,conn_list = list, min=0.1, max=0.4, delay = 1)
#sim1.connection_list_converter_modified_uniform(self_import = self_import,conn_list = list ,min_h = 0.9,max_h = 1.2,min_l = 0.6,max_l = 0.8,delay = 1)
sim1.start_sim(sim_time)
#sim1.display_weight()
sim1.plot_spikes("input",pre_pop_size , "Spike Pattern of Pre-Synaptic Population")
sim1.plot_spikes("output",post_pop_size, "Spike Pattern of Post-Synaptic Population")
#sim1.display_membrane_potential("Membrane Potential of one Post-Synaptic Neuron",xmin= 0,xmax = 400,ymin=-75,ymax = -57)
#sim1.display_membrane_potential2("Membrane Potential of Post-Synaptic Neuron",xmin= 0,xmax = 1200,ymin=-73,ymax = -60)
sim1.Plot_WeightDistribution(256,'Histogram of Trained Weight')
#sim1.Plot_WeightDistribution0(256,'Histogram of Trained Weight of neuron 0')
#sim1.Plot_WeightDistribution1(256,'Histogram of Trained Weight of neuron 1')