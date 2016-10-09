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
    post_pop_size = 20
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
        post_pop_size = 2
        STDP_mode =True
        inhibitory_spike_mode = False
        allsameweight = False
        self_import = False
        
#E_syn_weight = np.random.normal(0.2, 0.003,pre_pop_size*post_pop_size).tolist()
E_syn_weight = [0.2,0.1,0.2,0]
I_syn_weight = 3

setup_cond = {
                    'timestep': 1,
                    'min_delay':1,
                    'max_delay':144                    
} 

stdp_param = {
                    'tau_plus': 50.0,
                    'tau_minus': 150.0,
                    'w_min': 0,
                    'w_max': 1,
                    'A_plus': 0.01,
                    'A_minus': 0.05
}

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
    for j in range(num_copy):
        order_temp.append(random.randrange(0, 9))
    '''for j in range(num_copy):
        order_temp.append(random.randrange(9, 18))
    for j in range(num_copy):
        order_temp.append(random.randrange(18, 27))
    for j in range(num_copy):
        order_temp.append(random.randrange(27, 36))'''
    
    order = np.concatenate((order,order_temp))
    return order

#order = create_order(50)
    
order = create_shuffle_seq(12)
#order1 = range(27,36)
'''for i in range(50):
    order = np.concatenate((order,order1))'''

print "---------------------------------------------"
order = np.concatenate((order,last_episodes)) 
print order

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0]) #from us to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts


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
       
def raster_plot():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0,len(argw)):
        pylab.plot(argw[ii][3]+200*ii,argw[ii][0]+argw[ii][1]*16,".")
    pylab.title('raster plot of Virtual Retina Neuron Population')
    pylab.show()

raster_plot()
    
def BuildTrainingSpike(order,ONOFF):
    complete_Time = []
    for nid in range(0,pre_pop_size):
        SpikeTimes = []
        for tid in range(0,len(order)):
                temp=[]
                loc = order[tid]
                j = np.repeat(nid,len(argw[loc][1]))
                p = np.repeat(ONOFF,len(argw[loc][1]))
                temp = 200*tid+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]
                '''if (nid == 5) and (tid == 15):
                    print j,p,temp,len(j),len(p),len(argw[loc][1]),argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]'''
                if temp.size>0:
                    '''print j,p,temp,len(j),len(p),len(argw[loc][1]),argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]'''
                    SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
            complete_Time.append(SpikeTimes.tolist())
        else:
            complete_Time.append([])
    return complete_Time
    
#in_spike = BuildTrainingSpike(order,1)
#--------------------------------------------------# 

if(test_STDP):
    sim_time = 1000
    if(inhibitory_mode):
        in_spike = [[10],[],[],[]]
    else:
        #in_spike = [[0,10],[5],[20],[13]]
        #in_spike = [[0,10,100,110],[5,105],[20,120],[13,113]]
        #in_spike = [[0,10,100,110,200,210],[5,105,205],[20,120,220],[13,113,213]]
        order = [1,10]
        in_spike = BuildTrainingSpike(order,1)
else:
    in_spike = BuildTrainingSpike(order,1)
    sim_time = len(order)*animation_time
    
sim1.input_spike(in_spike)
list = [(0, 0, 0.2, 1), (1, 0, 0.1, 1), (2, 0, 0.25, 1), (3, 0, 0.3, 1)]
#sim1.connection_list_converter_normal(self_import = self_import,conn_list = list, mean=0.01, var=0.001,delay = 1)
sim1.connection_list_converter_uniform(self_import = self_import,conn_list = list, min=0.02, max=0.12, delay = 1)
#sim1.connection_list_converter_modified_uniform(self_import = self_import,conn_list = list ,min_h = 0.9,max_h = 1.2,min_l = 0.6,max_l = 0.8,delay = 1)
sim1.start_sim(sim_time)
#sim1.display_weight()
sim1.plot_spikes("input",pre_pop_size , "Spike Pattern of Pre-Synaptic Population")
sim1.plot_spikes("output",post_pop_size, "Spike Pattern of Post-Synaptic Population")
sim1.display_membrane_potential("Membrane Potential of Post-Synaptic Neuron",xmin= 0,xmax = 500,ymin=-75,ymax = -57)
sim1.Plot_WeightDistribution(256,'Histogram of Trained Weight')