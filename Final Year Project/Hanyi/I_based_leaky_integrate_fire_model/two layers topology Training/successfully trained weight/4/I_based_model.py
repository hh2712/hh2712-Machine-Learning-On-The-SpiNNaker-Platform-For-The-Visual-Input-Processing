__author__ = "hanyi"
import spynnaker.pyNN as sim
import pylab
import numpy as np
import random
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from types import FunctionType
import scipy.io

class IF_curr_exp_s:
    def __init__(self,pre_pop_size = None,post_pop_size = None,E_syn_weight= None,I_syn_weight = None,cell_params_lif = None,
        setup_cond = None,stdp_param = None,STDP_mode = True,inhibitory_spike_mode = False,allsameweight=False):
        #print "hello"
        #initialise all parameters for simulation
        self.pre_pop_size = pre_pop_size
        self.post_pop_size = post_pop_size
        self.E_syn_weight = E_syn_weight
        self.I_syn_weight = I_syn_weight
        self.cell_params_lif = cell_params_lif
        self.setup_cond = setup_cond
        self.stdp_param = stdp_param
        self.STDP_mode = STDP_mode
        self.inhibitory_spike_mode = inhibitory_spike_mode
        self.allsameweight = allsameweight
        
    def input_spike(self,in_spike):
        #construct own spike for now
        self.in_spike = in_spike;
        #print self.in_spike
    
    def connection_list_converter_normal(self,self_import,conn_list,mean,var,delay):
        if(self_import):
            self.conn_list = conn_list
        else:
            conn_list = []
            #self.init_weights = [[None]*self.post_pop_size]*self.pre_pop_size
            for i in range(self.post_pop_size):
                for j in range(self.pre_pop_size):
                    rand_num = np.random.normal(mean,var,1)
                    if(rand_num[0]<=0):
                        rand_num[0] = -rand_num[0]
                    if(rand_num[0]<0.1):
                        rand_num[0] = 0.1
                    conn_list.append((j,i,rand_num[0],delay))
            self.conn_list = conn_list
            '''for k in range(self.post_pop_size*self.pre_pop_size):
                self.init_weights[j][i] ='''
            
    
    def connection_list_converter_uniform(self,self_import,conn_list,min,max,delay):
        print "------------------------------------------------------------------"
        print "uniform distribution used"
        print "------------------------------------------------------------------"
        
        if(self_import):
            self.conn_list = conn_list
        else:
            conn_list = []
            #self.init_weights = [[None]*self.post_pop_size]*self.pre_pop_size
            for i in range(self.post_pop_size):
                for j in range(self.pre_pop_size):
                    rand_num = np.random.uniform(min,max,1)
                    if(rand_num[0]<0.05):
                        rand_num[0] = 0.05
                    conn_list.append((j,i,rand_num[0],delay))
            self.conn_list = conn_list
            '''for k in range(self.post_pop_size*self.pre_pop_size):
                self.init_weights[j][i] =''' 
            #print self.conn_list
            
    def connection_list_converter_modified_uniform(self,self_import,conn_list,min_h,max_h,min_l,max_l,delay):
        if(self_import):
            self.conn_list = conn_list
        else:
            conn_list = []
            for i in range(self.post_pop_size):
                for j in range(self.pre_pop_size):
                    if(i<5):
                        #x[0:4] y[0:4] 
                        if((j/16)<5 and (j%16)<5):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=5) and i<10):
                        #x[5:9] y[0:4]
                        if((j/16)>=5 and (j/16)<10  and (j%16)<5):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=10) and i<15):
                        #x[10:15] y[0:4]
                        if((j/16)>=10 and (j/16)<16 and (j%16)<5):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=15) and i<20):
                        #x[0:4] y[5:9]
                        if((j/16)<5 and (j%16)>=5 and (j%16)<10):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=20) and i<25):
                        #x[5:9] y[5:9]
                        if((j/16)>=5 and (j/16)<10 and (j%16)>=5 and (j%16)<10):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=25) and i<30):
                        #x[10:15] y[5:9]
                        if((j/16)>=10 and (j/16)<16 and (j%16)>=5 and (j%16)<10):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=30) and i<35):
                        #x[0:4] y[10:15]
                        if((j/16)<5 and (j%16)>=10 and (j%16)<16):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=35) and i<40):
                        #x[5:9] y[10:15]
                        if((j/16)>=5 and (j/16)<10 and (j%16)>=10 and (j%16)<16):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    if((i>=40) and i<45):
                        #x[10:15] y[10:15]
                        if((j/16)>=10 and (j/16)<16 and (j%16)>=10 and (j%16)<16):
                            rand_num = np.random.uniform(min_h,max_h,1)
                        else:
                            rand_num = np.random.uniform(min_l,max_l,1)
                    conn_list.append((j,i,rand_num[0],delay))
            self.conn_list = conn_list
    
    
    def start_sim(self,sim_time):
        #simulation setup
        self.simtime = sim_time
        sim.setup(timestep=self.setup_cond["timestep"], min_delay=self.setup_cond["min_delay"], max_delay=self.setup_cond["max_delay"])
        #initialise the neuron population
        spikeArrayOn = {'spike_times': self.in_spike}
        pre_pop = sim.Population(self.pre_pop_size, sim.SpikeSourceArray,
                        spikeArrayOn, label='inputSpikes_On')
        post_pop= sim.Population(self.post_pop_size,sim.IF_curr_exp,
                         self.cell_params_lif, label='post_1')
        stdp_model = sim.STDPMechanism(timing_dependence=sim.SpikePairRule(tau_plus= self.stdp_param["tau_plus"],
                                                    tau_minus= self.stdp_param["tau_minus"],
                                                    nearest=True),
                                        weight_dependence=sim.MultiplicativeWeightDependence(w_min= self.stdp_param["w_min"],
                                                               w_max= self.stdp_param["w_max"],
                                                               A_plus= self.stdp_param["A_plus"],
                                                               A_minus= self.stdp_param["A_minus"]))
        #initialise connectiviity of neurons
        #exitatory connection between pre-synaptic and post-synaptic neuron population                                                              
        if(self.inhibitory_spike_mode):
            connectionsOn  = sim.Projection(pre_pop, post_pop, sim.AllToAllConnector(weights = self.I_syn_weight,delays=1,
                                            allow_self_connections=False),
                                            target='inhibitory')
        else: 
            if(self.STDP_mode):
                if(self.allsameweight):
                    connectionsOn = sim.Projection(pre_pop, post_pop, 
                                               sim.AllToAllConnector(weights = self.E_syn_weight,delays=1),
                                                synapse_dynamics=sim.SynapseDynamics(slow=stdp_model),target='excitatory')
                else:
                    connectionsOn = sim.Projection(pre_pop, post_pop, 
                                               sim.FromListConnector(self.conn_list),
                                                synapse_dynamics=sim.SynapseDynamics(slow=stdp_model),target='excitatory')
            else:
                if(self.allsameweight):
                    connectionsOn  = sim.Projection(pre_pop, post_pop, sim.AllToAllConnector(weights = self.E_syn_weight,delays=1),
                                                                         target='excitatory')
                else:
                    connectionsOn  = sim.Projection(pre_pop, post_pop, sim.FromListConnector(self.conn_list),
                                                                         target='excitatory')
                #sim.Projection.setWeights(self.E_syn_weight)
        
        #inhibitory between the neurons post-synaptic neuron population
        connection_I  = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(weights = self.I_syn_weight,delays=1,
                                                             allow_self_connections=False),
                                                             target='inhibitory')
        pre_pop.record()                                                     
        post_pop.record()
        post_pop.record_v()
        sim.run(self.simtime)
        self.pre_spikes = pre_pop.getSpikes(compatible_output=True)
        self.post_spikes = post_pop.getSpikes(compatible_output=True)
        self.post_spikes_v = post_pop.get_v(compatible_output=True)
        self.trained_weights = connectionsOn.getWeights(format='array')
        sim.end()
        #print self.conn_list
        #print self.trained_weights
        '''scipy.io.savemat('trained_weight.mat',{'initial_weight':self.init_weights,
                                               'trained_weight':self.trained_weights
                                       })'''
        scipy.io.savemat('trained_weight.mat',{'trained_weight':self.trained_weights
                                       })
        
    def display_weight(self):
        for i in range(self.post_pop_size):
            print ([x[i] for x in self.weights_trained])
    
    def plot_spikes(self, spike_type, size, title):
        if (spike_type == "input"):
            spikes = self.pre_spikes
        if (spike_type == "output"):
            spikes = self.post_spikes
            
        if spikes is not None:
            pylab.figure()
            ax = plt.subplot(111, xlabel='Time/ms', ylabel='Neruons #',
                             title=title)
            pylab.xlim((0, self.simtime))
            pylab.ylim((0, size+2))
            lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],".")
            pylab.setp(lines,markersize=10,color='r')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
                 item.set_fontsize(20)
            pylab.show()
        else:
            print "No spikes received"
   
    
    def Plot_WeightDistribution(self,bin_num,title):
        hist,bins = np.histogram(self.trained_weights,bins=bin_num)
        center = (bins[:-1]+bins[1:])/2
        width = (bins[1]-bins[0])*0.7
        ax = pylab.subplot(111,xlabel='Weight',title =title)
        plt.bar(center,hist,align='center',width =width)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        plt.show()
    
    def display_membrane_potential(self,title,xmin=0,xmax=50,ymin=-70,ymax=-63): 
        post_spikes_v = self.post_spikes_v
        #print post_spikes_v
        if post_spikes_v is not None:
            pylab.figure()
            ax = plt.subplot(111, xlabel='Time/ms', ylabel=' Membrane Potential/V',
                             title=title)
            pylab.xlim((xmin, xmax))
            pylab.ylim((ymin, ymax))
            pylab.plot([i[1] for i in post_spikes_v], [i[2] for i in post_spikes_v])
            #pylab.setp(lines,markersize=10,color='r')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
                 item.set_fontsize(20)
            pylab.show()
        else:
            print "No spikes received"