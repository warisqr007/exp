## import random
import numpy as np
import pandas as pd

from config import *
from net_ import *
from alg_f import *


# init all networks
def init_nets():
    nets = []
    net = []
    if NET_NSF:
        for i in range(0,6):
            net.append(NationalScienceFoundation(NET_NUM_CHANNELS, NET_CHANNEL_FREE, NET_CHANNEL_BIAS))
        nets.append(net)
    
    net = []
    if NET_ARPA:
        for i in range(0,6):
            net.append(AdvancedResearchProjectsAgency(NET_NUM_CHANNELS, NET_CHANNEL_FREE, NET_CHANNEL_BIAS)) 
        nets.append(net)
        
    net = []
    if NET_CLARA:
        for i in range(0,6):
            net.append(CooperacionLatinoAmericana(NET_NUM_CHANNELS, NET_CHANNEL_FREE, NET_CHANNEL_BIAS))
        nets.append(net)
        
    net = []
    if NET_ITA:
        for i in range(0,6):
            net.append(Italian(NET_NUM_CHANNELS, NET_CHANNEL_FREE, NET_CHANNEL_BIAS))
        nets.append(net)
    
    net = []
    if NET_JANET:
        for i in range(0,6):
            net.append(JointAcademic(NET_NUM_CHANNELS, NET_CHANNEL_FREE, NET_CHANNEL_BIAS))
        nets.append(net)
    
    net = []
    if NET_RNP:
        for i in range(0,6):
            net.append(RedeNacionalPesquisa(NET_NUM_CHANNELS, NET_CHANNEL_FREE, NET_CHANNEL_BIAS))
        nets.append(net)

    return nets

# init all algorithms
def init_algs():
    algs = []
    if ALG_DFF:
        algs.append(DijkstraFirstFit())
    
    if ALG_DFF_G:
        algs.append(DijkstraFirstFitWithGrooming())
        
    if ALG_GA:
        algs.append(GeneticAlgorithm(GA_SIZE_POP, # 4 GA
                                        GA_MIN_GEN, GA_MAX_GEN,
                                        GA_MIN_CROSS_RATE, GA_MAX_CROSS_RATE,
                                        GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                                        GA_GEN_INTERVAL))
    if ALG_GA_G:
        algs.append(GeneticAlgorithmWithGrooming(GA_SIZE_POP, # 4 GA
                                        GA_MIN_GEN, GA_MAX_GEN,
                                        GA_MIN_CROSS_RATE, GA_MAX_CROSS_RATE,
                                        GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                                        GA_GEN_INTERVAL))    
    
    if ALG_FFGA:
        algs.append(FireFlyGAAlgorithm(FF_NUM,
                                     FF_MIN_GEN, FF_MAX_GEN,
                                     FF_MIN_CROSS_RATE, FF_MAX_CROSS_RATE,
                                     GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                                     GA_GEN_INTERVAL,absorption_coefficient_gamma))
        
    if ALG_FFGA_G:
        algs.append(FireFlyGAAlgorithmWithGrooming(FF_NUM,
                                     FF_MIN_GEN, FF_MAX_GEN,
                                     FF_MIN_CROSS_RATE, FF_MAX_CROSS_RATE,
                                     GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                                     GA_GEN_INTERVAL,absorption_coefficient_gamma))
        

    return algs

def poisson_arrival():
    r = np.random.uniform() 
    while r == 0.0 or r == 1.0:
        r = np.random.uniform()
    return -np.log(1-r)

def gen_od_pair(net):
    if not net.allow_multi_od:
        return net.source_node, net.dest_node

    # randomly choose origin and destination nodes
    origin      = random.randrange(net.num_nodes) 
    destination = random.randrange(net.num_nodes)

    # avoid origin node being the same as destination
    while origin == destination:
        destination = random.randrange(net.num_nodes) 

    return origin, destination

#calculate average connection request arrival rate
def util(load, NUM_CALLS):
    time=0
    for call in range(NUM_CALLS):
        until_next = poisson_arrival()/load
        time+=until_next
    
    return NUM_CALLS/time



nets = init_nets()
algs = init_algs()

if nets == [] or algs == []:
    sys.stderr.write('Something must be wrong. ')
    sys.stderr.flush()
    sys.exit(1)
    
for net in nets:
    for n in net:
        n.init_network()
    
# define the load in Erlangs
#for load in range(SIM_MIN_LOAD, SIM_MAX_LOAD):
load = 200
print('****************************************************************')
print('Load = %d ' %load)
# reset all networks to the inital state


for net in nets:
    for n in net:
        n.reset_network()
    # init the block array for each topology, considering each and every
    # node of the topology as an eventual origin node
    for a in algs:
        a.block_count[net[0].name] = []
        a.block_dict[net[0].name]  = {}
        for node in range(net[0].num_nodes):              # e.g.: cur node = 2
            a.block_count[net[0].name].append(0)          # list['nsf'][2] = 0
            a.block_dict[net[0].name][node] = np.empty(0) # dict['nsf'][2] = 0.0
            #a.traffic_arrival[n.name].append(0)

#result = pd.DataFrame(columns=['Network', 'Algorithm', 'SIM_NUM_CALLS','Block_count', 'Blocking_Probability', 'Traffic'])      
result = []    


for net in nets:
    i=0
    time=0
    print('************************************************')
    print(net[0].name)
    # call requests arrival
    avg_arrival_rate=0
    avg_holding_time=0
    for call in range(SIM_NUM_CALLS):
        # exponential time distributions
        holding_time = poisson_arrival() # define a time for a λ to be released
        until_next   = poisson_arrival()/load # define interarrival times

        # FIXME apply RWA
        o, d = gen_od_pair(net[0])
        cr = random.randint(5,21)/100
        for a in algs:
            (t,route,wavelength,groomed) = a.rwa(net[i%6], o, d, holding_time,cr,time)
            i+=1
            a.block_count[net[0].name][o] += t
            #a.traffic_arrival[n.name][o] +=(holding_time/until_next)
            #avg_holding_time = ((avg_holding_time*call) + holding_time)/(call+1)
            #avg_arrival_rate = ((avg_arrival_rate*call) + 1/until_next)/(call+1)
            #print('***********************************')
            #print('Origin %d, Destination %d ' % (o,d))
            #print('Route :')
            #print(route)
            #print('Wavelength : %d' %wavelength)

        # update networks
        for n in net:
            n.update_network(until_next)
        # exits Poisson for (call) loop
        time+=until_next

    for a in algs:
        print('*******************')
        print(a.name)
        print('Block Count : ')
        print(a.block_count[net[0].name])
        sum_block_count = sum(a.block_count[net[0].name])
        blocking_probability = (sum_block_count/SIM_NUM_CALLS)
        print('Blocking Probability : %f' %blocking_probability)
        #traffic = avg_arrival_rate*avg_holding_time
        #print('Traffic : %f' %traffic)
        result.append({'Network' : net[0].name, 'Algorithm' : a.name, 'SIM_NUM_CALLS' : SIM_NUM_CALLS, 'Block_count' : sum_block_count, 'Blocking_Probability' : blocking_probability})

rs = pd.DataFrame(result, columns=['Network', 'Algorithm', 'SIM_NUM_CALLS','Block_count', 'Blocking_Probability'])

rs.to_csv('Result_calls200.csv')
print('Done')
    