import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random

class FireFlyEnvironment(object):
    def __init__(self):
        pass
    
    # generates initial population with random but valid chromosomes
    def generate_fireflies(self, A, W, num_nodes, num_channels, o, d):
        fireflies = [] # [ [[route], [L], wl_avail, r_len], ..., ]
        trials = 0
        while len(fireflies) < self.FF_NUM and trials < 300:
            nodes = list(range(num_nodes)) # router indexes
            route = self.generate_firefly_route(A, W, num_nodes, num_channels, o, d, nodes)
            firefly = [route, [], 0, 0]
            if route and firefly not in fireflies:
                fireflies.append(firefly)
                trials = 0
            else:
                trials += 1
            pass
        return fireflies

    # TODO: Create Population
    def generate_firefly_route(self, adj_mtx, wave_mtx, num_nodes, num_channels, start_router, end_router, nodes):
        
        # 1. start from source node
        current_router = start_router
        route = [nodes.pop(nodes.index(current_router))]
        while len(nodes):
            # 2. randomly choose, with equal probability, one of the nodes that
            # is SURELY connected to the current node to be the next in path
            neighbour_routers = []
            for router in nodes:
                if adj_mtx[current_router][router]:
                    neighbour_routers.append([router, 0])
                    
            if not neighbour_routers:
                route = False
                break
            else:
                for ind in len(neighbour_routers):
                    router = neighbour_routers[ind][0]
                    neighbour_routers[ind][1] = self.evaluate_attractiveness(wave_mtx, num_channels, current_router, router)
            
                current_router = self.most_attractive(neighbour_routers)
                route.append(nodes.pop(nodes.index(current_router)))
                
                if current_router == end_router:
                    break

        if route and len(route) > num_nodes:
            route = False

        return route
    
    ##TO_DO
    def evaluate_attractiveness(self, wave_mtx, num_channels, current_router, router):
        avail_w = 0
        for w in range(num_channels):
            # if the wavelength is available at the output node
            if wave_mtx[current_router][router][w]==1:
                avail_w +=1
        
        attractiveness_beta_at_distance_0 = 1
        distance_of_two_fireflies = 1/(wl_avail+0.0000001)
        attractiveness_beta = attractiveness_beta_at_distance_0 * math.exp(-self.absorption_coefficient_gamma * distance_of_two_fireflies**2)
                
        return attractiveness_beta
    
    ##To-Do
    def most_attractive(self, neighbour_routers):
        return neighbour_routers[0][0]

    def evaluate(self, net, route):
        l = len(route)-1
        L = [] # labels
        for w in range(net.num_channels):
            num = 0
            for i in range(l):
                rcurr = route[i]
                rnext = route[i+1]
                num += (w+1) * net.wave_mtx[rcurr][rnext][w]
            L.append(num/float((w+1)*l))

        wl_avail = 0
        for label in L:
            if label == 1.0:
                wl_avail += 1
                

        return L, wl_avail, l+1 # Label, λ's available, length of route and attractiveness
    

    # [[chrom], [L], wl_avail, r_len]
    def insertion_sort(self, A):
        for j in range(1, len(A)):
            R = A[j]

            i = j-1
            if R[2]: # if route have λ available, then R[2] > 0
                while i >= 0 and A[i][3] > R[3]:
                    A[i+1] = A[i]
                    i -= 1
                A[i+1] = R

        return A

### EOF ###