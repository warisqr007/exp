import sys
import os

import numpy as np
import networkx as nx
import itertools
import random
import operator

from ga_env import *
from fireflyGA_env import *
from firefly_env import *


class Routing(object):
    """ Class Routing """

    def __init__(self):
        pass

    def is_od_pair_ok(self, adj_mtx, orig, dest):
        if orig == dest:
            return False
        # TODO those two 'elif's below are really really necessary?
        elif orig < 0 or dest < 0: 
            sys.stderr.write('Invalid\n')
            sys.stderr.flush()
            return False
        elif orig > adj_mtx.shape[0] or dest > adj_mtx.shape[0]:
            sys.stderr.write('Invalid\n')
            sys.stderr.flush()
            return False
        else:
            return True

        
    def dijkstra(self, A, o, d):
        if not self.is_od_pair_ok(A, o, d):
            sys.stderr.write('Error: source (%d) or destination (%d) ' % (o,d))
            sys.stderr.write('indexes are invalid.\n')
            sys.stderr.flush()
            return None

        G = nx.from_numpy_matrix(A, create_using=nx.Graph())
        hops, path = nx.bidirectional_dijkstra(G, o, d, weight=None)
        return path
    

class WavelengthAssignment(object):
    """ Class Wavelength Assignment """

    def __init__(self):
        pass

    # local knowledge first fit wavelength assignment
    def first_fit(self, W, route, num_channels):
        """ Certainly does something """

        # look at the source node only (1st link)
        rcurr = route[0] 
        rnext = route[1]
        # Check whether each wavelength w is available 
        # on the first link of route R (only at the 1st link)
        avail_w = []
        for w in range(num_channels):
            # if the wavelength is available at the output node
            if W[rcurr][rnext][w]==1:
                # return it as the first one who fits :)
                avail_w.append(w)

        return avail_w # no wavelength available at the output of the source node

    def random(self):
        pass

    def most_used(self):
        pass

    def least_used(self):
        pass

class RWAAlgorithm(Routing, WavelengthAssignment):

    def __init__(self):
        super(RWAAlgorithm, self).__init__()
        #self.traffic_arrival = {}
        self.block_count = {} # to store the number of blocked calls
        self.block_dict  = {} # store percentage of blocked calls per generation

    def is_wave_available(self, wave_mtx, route, wavelength):
        """ check if a wavelength is available over all links of the path """

        # check if the λ chosen at the first link is availble on all links of R
        length = len(route)
        for r in range(length-1):
            rcurr = route[r]
            rnext = route[r+1]

            # if not available in any of the next links, block
            if wave_mtx[rcurr][rnext][wavelength]!=1:
                return False # call must be blocked

        return True
    
    def is_wave_available_for_grooming(self, wave_mtx, route, wavelength, cr):
        """ check if enough capacity in the wavelength is available over all links of the path to be groomed """
        # check if the λ chosen at the first link is availble on all links of R
        length = len(route)
        for r in range(length-1):
            rcurr = route[r]
            rnext = route[r+1]

            # if not available in any of the next links, block
            if wave_mtx[rcurr][rnext][wavelength] < cr:
                return False # call must be blocked

        return True

    def alloc_net_resources(self, net, route, o, d, w, ht, cr, gr):
        """ eita giovana """
        
        if not gr:
            # update traffic matrix 1/2: tuple dicts
            key = (o, d, w)
            if key not in list(net.traffic_mtx):
                net.traffic_mtx[key] = {}
        
            net.traffic_mtx[key].update({tuple(route):ht})
        
        # increase traffic matrix's call counter
        net.traffic_mtx['num_calls'] += 1 

        # update trafffic matrix 2/2: np 3D array
        length = len(route)
        for r in range(length-1):
            rcurr = route[r]
            rnext = route[r+1]

            # make λ NOT available on all links of the path 
            net.wave_mtx[rcurr][rnext][w] -= cr
            net.wave_mtx[rnext][rcurr][w] -= cr 
            
            if not gr:
                # assign a time for it to be free again (released)
                net.traffic_mtx['time'][rcurr][rnext][w] = ht
                net.traffic_mtx['time'][rnext][rcurr][w] = ht
                
                key_c = (rcurr, d, w)
                route_c = route[r:]
                if key_c not in list(net.conn_mtx):
                    net.conn_mtx[key_c] = {}
        
                net.conn_mtx[key_c].update({tuple(route_c):ht})
            
            
    def is_lightpath_existing(self, net, o, d, w, ht, cr):
        '''Checks for existing lightpath to groom with'''
        key = (o, d, w)
        if key in list(net.conn_mtx):
            for path in list(net.conn_mtx[key]):
                if net.conn_mtx[key][path] >= ht and self.is_wave_available_for_grooming(net.wave_mtx, path, w, cr):
                    return (True, path)
        
        return (False, None)
    
    
    def stateful_grooming(self, net, avail_w, o, d, ht, cr):
        for w in range(net.num_channels):
            if w not in avail_w:
                (avl, route) = self.is_lightpath_existing(net, o, d, w, ht, cr)
                if avl:
                    self.alloc_net_resources(net, route, o, d, w, ht, cr, 1)
                    return (1, route, w)
                
        return (0,None,-1)
                

    def save_erlang_blocks(self, net_key, net_num_nodes, total_calls):
        for node in range(net_num_nodes):
            # compute a percentage of blocking probability per Erlang
            bp_per_erlang = 100.0 * self.block_count[net_key][node] / total_calls 
            self.block_dict[net_key][node] = np.append(self.block_dict[net_key][node], bp_per_erlang)

    def plot_fits(self, fits, PT_BR=False):
        """ This method plots """

        if PT_BR:
            import sys
            reload(sys)  
            sys.setdefaultencoding('utf8') # for plot in PT_BR

        import matplotlib.pyplot as plt
        import matplotlib.animation as anim
        from matplotlib.ticker import EngFormatter

        # do not interrupt program flow to plot
        plt.interactive(True) 

        fig, ax = plt.subplots()
        ax.set_xscale('linear')

        formatter = EngFormatter(unit='', places=1)
        ax.xaxis.set_major_formatter(formatter)

        if PT_BR:
            ax.set_xlabel(u'Gerações', fontsize=20)
            ax.set_ylabel(u'Número de chamadas atendidas', fontsize=20)
        else:
            ax.set_xlabel(u'Generations', fontsize=20)
            ax.set_ylabel(u'Number of attended calls', fontsize=20)
    
        ax.grid()
        ax.plot(fits, linewidth=2.0)
    
        x = range(0, 0, 5) # FIXME
        y = np.arange(0, 8, 1)
    
        if PT_BR:
            title =  u'Melhores fits de %d indivíduos por geração' % 0
        else:
            title =  u'Best fit values of %d individuals per generation' % 0

        fig.suptitle(title, fontsize=20)

        #plt.margins(0.02)
        plt.subplots_adjust(bottom=0.12)

        #plt.xticks(x)
        plt.yticks(y)
        plt.draw()
        plt.show(block=True)

    # TODO
    def plot_bp(self, net):
        """ Plot blocing probabilities """

        if PT_BR:
            import sys
            reload(sys)  
            sys.setdefaultencoding('utf8') # for plot in PT_BR

        import matplotlib.pyplot as plt
        import matplotlib.animation as anim
        from matplotlib.ticker import EngFormatter

        # do not interrupt program flow to plot
        plt.interactive(True) 

        fig, ax = plt.subplots()
        ax.set_xscale('linear')

        formatter = EngFormatter(unit='', places=1)
        ax.xaxis.set_major_formatter(formatter)
        

class DijkstraFirstFit(RWAAlgorithm):
    """ Dijkstra and First Fit """

    def __init__(self):
        super(DijkstraFirstFit, self).__init__()
        self.name     = 'DFF'
        self.fullname = 'Dijkstra and First Fit'

    def rwa(self, net, orig, dest, hold_t, call_req):  # call_req : call request bandwidth requirement
        """ This method RWAs """
        # call the routing method
        route = self.dijkstra(net.adj_mtx, orig, dest)

        # call the wavelength assignment method
        avail_w = self.first_fit(net.wave_mtx, route, net.num_channels)

        groomed = 0  # To indicate if the wavelength is groomed
        # if WA was successful, allocate resources on the network
        if avail_w:
            for wavelength in avail_w:
                if self.is_wave_available(net.wave_mtx, route, wavelength):
                    self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
                    return (0,route,wavelength,groomed) # allocated

        return (1,route,-1,-1)

                
class DijkstraFirstFitWithGrooming(RWAAlgorithm):
    """ Dijkstra and First Fit with Grooming"""

    def __init__(self):
        super(DijkstraFirstFitWithGrooming, self).__init__()
        self.name     = 'DFF_G'
        self.fullname = 'Dijkstra and First Fit with Grooming'

    def rwa(self, net, orig, dest, hold_t, call_req):  # call_req : call request bandwidth requirement 
        """ This method RWAs """
        # call the routing method
        route = self.dijkstra(net.adj_mtx, orig, dest)

        # call the wavelength assignment method
        avail_w = self.first_fit(net.wave_mtx, route, net.num_channels)

        groomed = 0  # To indicate if the wavelength is groomed
        # if WA was successful, allocate resources on the network
        if avail_w:
            for wavelength in avail_w:
                if self.is_wave_available(net.wave_mtx, route, wavelength):
                    self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
                    return (0,route,wavelength,groomed) # allocated
        

        #Stateful Grooming
        (groomed,route,wavelength) = self.stateful_grooming(net, avail_w, orig, dest, hold_t, call_req)
        if groomed:
            return (0,route,wavelength,groomed)

        
        return (1,route,-1,-1)


    
##############################################################

# TODO: Main Genetic Algorithm Function
class GeneticAlgorithm(RWAAlgorithm, Environment):
    """ GA class """

    def __init__(self, GA_SIZE_POP,
                    GA_MIN_GEN, GA_MAX_GEN,
                    GA_MIN_CROSS_RATE, GA_MAX_CROSS_RATE,
                    GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                    GA_GEN_INTERVAL):
        super(GeneticAlgorithm, self).__init__()
        self.GA_SIZE_POP       = GA_SIZE_POP
        self.GA_MIN_GEN        = GA_MIN_GEN
        self.GA_MAX_GEN        = GA_MAX_GEN
        self.GA_MIN_CROSS_RATE = GA_MIN_CROSS_RATE
        self.GA_MAX_CROSS_RATE = GA_MAX_CROSS_RATE
        self.GA_MIN_MUT_RATE   = GA_MIN_MUT_RATE
        self.GA_MAX_MUT_RATE   = GA_MAX_MUT_RATE
        self.GA_GEN_INTERVAL   = GA_GEN_INTERVAL
        
        self.name     = 'GA'
        self.fullname = 'Genetic Algorithm'

    def rwa(self, net, orig, dest, hold_t, call_req):
        population = self.init_population(net.adj_mtx, net.num_nodes, orig, dest)
        fits = []
        stop_criteria = 0
        count = 0
        while not stop_criteria:
            # perform evaluation (fitness calculation)
            for ind in range(len(population)):
                L, wl_avail, r_len = self.evaluate(net, population[ind][0])
                population[ind][1] = L
                population[ind][2] = wl_avail
                population[ind][3] = r_len
            # perform selection
            mating_pool = self.select(list(population), self.GA_MAX_CROSS_RATE)
            # perform crossover
            offspring = self.cross(mating_pool)
            if offspring:
                for child in offspring:
                    population.pop()
                    population.insert(0, [child, [], 0, 0])

            # perform mutation
            for i in range(int(math.ceil(self.GA_MIN_MUT_RATE*len(population)))):
                normal_ind = random.choice(population)
                trans_ind = self.mutate(net.adj_mtx, normal_ind[0], net.num_nodes) # X MEN
                if trans_ind != normal_ind:
                    population.remove(normal_ind)
                    population.insert(0, [trans_ind, [], 0, 0])

            # sort population according to length .:. shortest paths first
            population.sort(key=operator.itemgetter(2), reverse=True)

            # sort population according to wavelength availability
            population = self.insertion_sort(population)
            
            count += 1
            if count>=self.GA_MIN_GEN:
                stop_criteria = 1

        # perform evaluation (fitness calculation)
        for ind in range(len(population)):
            L, wl_avail, r_len = self.evaluate(net, population[ind][0])
            population[ind][1]  = L
            population[ind][2]  = wl_avail
            population[ind][3]  = r_len

        # sort population according to length: shortest paths first
        population.sort(key=operator.itemgetter(2), reverse=True)

        # sort population according to wavelength availability
        population = self.insertion_sort(population)
        
        
        groomed = 0
        
        # update NSF graph
        route = population[0][0]
        if population[0][2] > 0:
            wavelength = population[0][1].index(1)
            self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
            return (0,route,wavelength,groomed) # allocated
        
        return (1,route,-1,-1) # blocked
    



class GeneticAlgorithmWithGrooming(RWAAlgorithm, Environment):
    """ GA class - Groomed """

    def __init__(self, GA_SIZE_POP,
                    GA_MIN_GEN, GA_MAX_GEN,
                    GA_MIN_CROSS_RATE, GA_MAX_CROSS_RATE,
                    GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                    GA_GEN_INTERVAL):
        super(GeneticAlgorithmWithGrooming, self).__init__()
        self.GA_SIZE_POP       = GA_SIZE_POP
        self.GA_MIN_GEN        = GA_MIN_GEN
        self.GA_MAX_GEN        = GA_MAX_GEN
        self.GA_MIN_CROSS_RATE = GA_MIN_CROSS_RATE
        self.GA_MAX_CROSS_RATE = GA_MAX_CROSS_RATE
        self.GA_MIN_MUT_RATE   = GA_MIN_MUT_RATE
        self.GA_MAX_MUT_RATE   = GA_MAX_MUT_RATE
        self.GA_GEN_INTERVAL   = GA_GEN_INTERVAL
        
        self.name     = 'GA_G'
        self.fullname = 'Genetic Algorithm with Grooming'

    def rwa(self, net, orig, dest, hold_t, call_req):
        population = self.init_population(net.adj_mtx, net.num_nodes, orig, dest)
        fits = []
        stop_criteria = 0
        count = 0
        while not stop_criteria:
            # perform evaluation (fitness calculation)
            for ind in range(len(population)):
                L, wl_avail, r_len = self.evaluate(net, population[ind][0])
                population[ind][1] = L
                population[ind][2] = wl_avail
                population[ind][3] = r_len
            # perform selection
            mating_pool = self.select(list(population), self.GA_MAX_CROSS_RATE)
            # perform crossover
            offspring = self.cross(mating_pool)
            if offspring:
                for child in offspring:
                    population.pop()
                    population.insert(0, [child, [], 0, 0])

            # perform mutation
            for i in range(int(math.ceil(self.GA_MIN_MUT_RATE*len(population)))):
                normal_ind = random.choice(population)
                trans_ind = self.mutate(net.adj_mtx, normal_ind[0], net.num_nodes) # X MEN
                if trans_ind != normal_ind:
                    population.remove(normal_ind)
                    population.insert(0, [trans_ind, [], 0, 0])

            # sort population according to length .:. shortest paths first
            population.sort(key=operator.itemgetter(2), reverse=True)

            # sort population according to wavelength availability
            population = self.insertion_sort(population)
            
            count += 1
            if count>=self.GA_MIN_GEN:
                stop_criteria = 1

        # perform evaluation (fitness calculation)
        for ind in range(len(population)):
            L, wl_avail, r_len = self.evaluate(net, population[ind][0])
            population[ind][1]  = L
            population[ind][2]  = wl_avail
            population[ind][3]  = r_len

        # sort population according to length: shortest paths first
        population.sort(key=operator.itemgetter(2), reverse=True)

        # sort population according to wavelength availability
        population = self.insertion_sort(population)
        
        
        groomed = 0
        
        # update NSF graph
        route = population[0][0]
        if population[0][2] > 0:
            wavelength = population[0][1].index(1)
            self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
            return (0,route,wavelength,groomed) # allocated
        
        avail_w = self.first_fit(net.wave_mtx, route, net.num_channels)
        
        #Stateful Grooming
        (groomed,route,wavelength) = self.stateful_grooming(net, avail_w, orig, dest, hold_t, call_req)
        if groomed:
            return (0,route,wavelength,groomed)

        return (1,route,-1,-1) # blocked
    
######################################################################


class FireFlyAlgorithm(RWAAlgorithm, FireFlyEnvironment):
    """ FireFly class """

    def __init__(self, FF_NUM,
                    FF_MIN_GEN, FF_MAX_GEN,
                    FF_MIN_CROSS_RATE, FF_MAX_CROSS_RATE,
                    GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                    GA_GEN_INTERVAL,absorption_coefficient_gamma):
        super(FireFlyAlgorithm, self).__init__()
        self.FF_NUM            = FF_NUM
        self.FF_MIN_GEN        = FF_MIN_GEN
        self.FF_MAX_GEN        = FF_MAX_GEN
        self.FF_MIN_CROSS_RATE = FF_MIN_CROSS_RATE
        self.FF_MAX_CROSS_RATE = FF_MAX_CROSS_RATE
        self.GA_MIN_MUT_RATE   = GA_MIN_MUT_RATE
        self.GA_MAX_MUT_RATE   = GA_MAX_MUT_RATE
        self.GA_GEN_INTERVAL   = GA_GEN_INTERVAL
        self.absorption_coefficient_gamma = absorption_coefficient_gamma
        
        self.name     = 'FF'
        self.fullname = 'FireFly Algorithm'

    def rwa(self, net, orig, dest, hold_t, call_req):
        fireflies = self.generate_fireflies(net.adj_mtx, net.wave_mtx, net.num_nodes, net.num_channels, orig, dest)
        
        
        # perform evaluation (fitness calculation)
        for ind in range(len(fireflies)):
            L, wl_avail, r_len = self.evaluate(net, fireflies[ind][0])
            fireflies[ind][1]  = L
            fireflies[ind][2]  = wl_avail
            fireflies[ind][3]  = r_len


        # sort population according to length: shortest paths first
        fireflies.sort(key=operator.itemgetter(2), reverse=True)

        # sort population according to wavelength availability
        fireflies = self.insertion_sort(fireflies)
        
        groomed = 0
        
        # update NSF graph
        route = fireflies[0][0]
        if fireflies[0][2] > 0:
            wavelength = fireflies[0][1].index(1)
            self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
            return (0,route,wavelength,groomed) # allocated
        
        return (1,route,-1,-1) # blocked

    
    
class FireFlyAlgorithmWithGrooming(RWAAlgorithm, FireFlyEnvironment):
    """ FireFly class - Groomed """

    def __init__(self, FF_NUM,
                    FF_MIN_GEN, FF_MAX_GEN,
                    FF_MIN_CROSS_RATE, FF_MAX_CROSS_RATE,
                    GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                    GA_GEN_INTERVAL,absorption_coefficient_gamma):
        super(FireFlyAlgorithmWithGrooming, self).__init__()
        self.FF_NUM            = FF_NUM
        self.FF_MIN_GEN        = FF_MIN_GEN
        self.FF_MAX_GEN        = FF_MAX_GEN
        self.FF_MIN_CROSS_RATE = FF_MIN_CROSS_RATE
        self.FF_MAX_CROSS_RATE = FF_MAX_CROSS_RATE
        self.GA_MIN_MUT_RATE   = GA_MIN_MUT_RATE
        self.GA_MAX_MUT_RATE   = GA_MAX_MUT_RATE
        self.GA_GEN_INTERVAL   = GA_GEN_INTERVAL
        self.absorption_coefficient_gamma = absorption_coefficient_gamma
        
        self.name     = 'FF_G'
        self.fullname = 'FireFly Algorithm with Grooming'

    def rwa(self, net, orig, dest, hold_t, call_req):
        fireflies = self.generate_fireflies(net.adj_mtx, net.wave_mtx, net.num_nodes, orig, dest)
        
        
        # perform evaluation (fitness calculation)
        for ind in range(len(fireflies)):
            L, wl_avail, r_len = self.evaluate(net, fireflies[ind][0])
            fireflies[ind][1]  = L
            fireflies[ind][2]  = wl_avail
            fireflies[ind][3]  = r_len


        # sort population according to length: shortest paths first
        fireflies.sort(key=operator.itemgetter(2), reverse=True)

        # sort population according to wavelength availability
        fireflies = self.insertion_sort(fireflies)
        
        groomed = 0
        
        # update NSF graph
        route = fireflies[0][0]
        if fireflies[0][2] > 0:
            wavelength = fireflies[0][1].index(1)
            self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
            return (0,route,wavelength,groomed) # allocated
        
        
        avail_w = self.first_fit(net.wave_mtx, route, net.num_channels)
        
        #Stateful Grooming
        (groomed,route,wavelength) = self.stateful_grooming(net, avail_w, orig, dest, hold_t, call_req)
        if groomed:
            return (0,route,wavelength,groomed)

        return (1,route,-1,-1) # blocked
        


#######################################################################


class FireFlyGAAlgorithm(RWAAlgorithm, FireFlyGAEnvironment):
    """ FireFly class """

    def __init__(self, FF_NUM,
                    FF_MIN_GEN, FF_MAX_GEN,
                    FF_MIN_CROSS_RATE, FF_MAX_CROSS_RATE,
                    GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                    GA_GEN_INTERVAL,absorption_coefficient_gamma):
        super(FireFlyGAAlgorithm, self).__init__()
        self.FF_NUM            = FF_NUM
        self.FF_MIN_GEN        = FF_MIN_GEN
        self.FF_MAX_GEN        = FF_MAX_GEN
        self.FF_MIN_CROSS_RATE = FF_MIN_CROSS_RATE
        self.FF_MAX_CROSS_RATE = FF_MAX_CROSS_RATE
        self.GA_MIN_MUT_RATE   = GA_MIN_MUT_RATE
        self.GA_MAX_MUT_RATE   = GA_MAX_MUT_RATE
        self.GA_GEN_INTERVAL   = GA_GEN_INTERVAL
        self.absorption_coefficient_gamma = absorption_coefficient_gamma
        
        self.name     = 'FF-GA'
        self.fullname = 'FireFly-GA Algorithm'

    def rwa(self, net, orig, dest, hold_t, call_req):
        fireflies = self.init_fireflies(net.adj_mtx, net.num_nodes, orig, dest)
        fits = []
        stop_criteria = 0
        count = 0
        while not stop_criteria:
            # perform evaluation (fitness calculation)
            for firefly in range(len(fireflies)):
                L, wl_avail, r_len, attractiveness_beta = self.evaluate(net, fireflies[firefly][0])
                fireflies[firefly][1] = L
                fireflies[firefly][2] = wl_avail
                fireflies[firefly][3] = r_len
                fireflies[firefly][4] = attractiveness_beta
                
            children = []
               
            for firefly_i in range(len(fireflies)):
                for firefly_j in range(len(fireflies)):
                    if fireflies[firefly_i][4] < fireflies[firefly_j][4]:
                        mating_pool = [fireflies[firefly_i][0],fireflies[firefly_j][0]]
                        offspring = self.cross(mating_pool)
                        if offspring:
                            for child in offspring:
                                if child and [child, [], 0, 0, 0] not in children:
                                    children.append([child, [], 0, 0, 0])
                

            
            fireflies = fireflies + children

            # sort population according to length .:. shortest paths first
            fireflies.sort(key=operator.itemgetter(2), reverse=True)

            # sort population according to wavelength availability
            fireflies = self.insertion_sort(fireflies)
            
            fireflies = fireflies[:self.FF_NUM]
            
            count += 1
            if count>=self.FF_MIN_GEN:
                stop_criteria = 1

        # perform evaluation (fitness calculation)
        for firefly in range(len(fireflies)):
            L, wl_avail, r_len, attractiveness_beta = self.evaluate(net, fireflies[firefly][0])
            fireflies[firefly][1]  = L
            fireflies[firefly][2]  = wl_avail
            fireflies[firefly][3]  = r_len
            fireflies[firefly][4]  = attractiveness_beta

        # sort population according to length: shortest paths first
        fireflies.sort(key=operator.itemgetter(2), reverse=True)

        # sort population according to wavelength availability
        fireflies = self.insertion_sort(fireflies)
        
        groomed = 0
        
        # update NSF graph
        route = fireflies[0][0]
        if fireflies[0][2] > 0:
            wavelength = fireflies[0][1].index(1)
            self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
            return (0,route,wavelength,groomed) # allocated
        
        return (1,route,-1,-1) # blocked
    
    
    


class FireFlyGAAlgorithmWithGrooming(RWAAlgorithm, FireFlyGAEnvironment):
    """ FireFly class - Groomed """

    def __init__(self, FF_NUM,
                    FF_MIN_GEN, FF_MAX_GEN,
                    FF_MIN_CROSS_RATE, FF_MAX_CROSS_RATE,
                    GA_MIN_MUT_RATE, GA_MAX_MUT_RATE,
                    GA_GEN_INTERVAL,absorption_coefficient_gamma):
        super(FireFlyGAAlgorithmWithGrooming, self).__init__()
        self.FF_NUM            = FF_NUM
        self.FF_MIN_GEN        = FF_MIN_GEN
        self.FF_MAX_GEN        = FF_MAX_GEN
        self.FF_MIN_CROSS_RATE = FF_MIN_CROSS_RATE
        self.FF_MAX_CROSS_RATE = FF_MAX_CROSS_RATE
        self.GA_MIN_MUT_RATE   = GA_MIN_MUT_RATE
        self.GA_MAX_MUT_RATE   = GA_MAX_MUT_RATE
        self.GA_GEN_INTERVAL   = GA_GEN_INTERVAL
        self.absorption_coefficient_gamma = absorption_coefficient_gamma
        
        self.name     = 'FF-GA_G'
        self.fullname = 'FireFly-GA Algorithm with Grooming'

    def rwa(self, net, orig, dest, hold_t, call_req):
        fireflies = self.init_fireflies(net.adj_mtx, net.num_nodes, orig, dest)
        fits = []
        stop_criteria = 0
        count = 0
        while not stop_criteria:
            # perform evaluation (fitness calculation)
            for firefly in range(len(fireflies)):
                L, wl_avail, r_len, attractiveness_beta = self.evaluate(net, fireflies[firefly][0])
                fireflies[firefly][1] = L
                fireflies[firefly][2] = wl_avail
                fireflies[firefly][3] = r_len
                fireflies[firefly][4] = attractiveness_beta
                
            children = []
               
            for firefly_i in range(len(fireflies)):
                for firefly_j in range(len(fireflies)):
                    if fireflies[firefly_i][4] < fireflies[firefly_j][4]:
                        mating_pool = [fireflies[firefly_i][0],fireflies[firefly_j][0]]
                        offspring = self.cross(mating_pool)
                        if offspring:
                            for child in offspring:
                                if child and [child, [], 0, 0, 0] not in children:
                                    children.append([child, [], 0, 0, 0])
             
            
            fireflies = fireflies + children

            # sort population according to length .:. shortest paths first
            fireflies.sort(key=operator.itemgetter(2), reverse=True)

            # sort population according to wavelength availability
            fireflies = self.insertion_sort(fireflies)
            
            fireflies = fireflies[:self.FF_NUM]
            
            count += 1
            if count>=self.FF_MIN_GEN:
                stop_criteria = 1

        # perform evaluation (fitness calculation)
        for firefly in range(len(fireflies)):
            L, wl_avail, r_len, attractiveness_beta = self.evaluate(net, fireflies[firefly][0])
            fireflies[firefly][1]  = L
            fireflies[firefly][2]  = wl_avail
            fireflies[firefly][3]  = r_len
            fireflies[firefly][4]  = attractiveness_beta

        # sort population according to length: shortest paths first
        fireflies.sort(key=operator.itemgetter(2), reverse=True)

        # sort population according to wavelength availability
        fireflies = self.insertion_sort(fireflies)
        
        groomed = 0
        
        # update NSF graph
        route = fireflies[0][0]
        if fireflies[0][2] > 0:
            wavelength = fireflies[0][1].index(1)
            self.alloc_net_resources(net, route, orig, dest, wavelength, hold_t, call_req, groomed)
            return (0,route,wavelength,groomed) # allocated    
        
        avail_w = self.first_fit(net.wave_mtx, route, net.num_channels)
        
        #Stateful Grooming
        (groomed,route,wavelength) = self.stateful_grooming(net, avail_w, orig, dest, hold_t, call_req)
        if groomed:
            return (0,route,wavelength,groomed)

        return (1,route,-1,-1) # blocked
    



### EOF ###


