import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random

class FireFlyGAEnvironment(object):
    def __init__(self):
        pass

    # generates initial population with random but valid chromosomes
    def init_fireflies(self, A, num_nodes, o, d):
        fireflies = [] # [ [[chrom], [L], wl_avail, r_len], ..., ]
        trials = 0
        while len(fireflies) < self.FF_NUM and trials < 300:
            nodes = list(range(num_nodes)) # router indexes
            route = self.generate_firefly_route(A, num_nodes, o, d, nodes)
            firefly = [route, [], 0, 0, 0]
            if route and firefly not in fireflies:
                fireflies.append(firefly)
                trials = 0
            else:
                trials += 1
            pass
        return fireflies

    # TODO: Create Population
    def generate_firefly_route(self, adj_mtx, num_nodes, start_router, end_router, nodes):
        count = 0
        # 1. start from source node
        current_router = start_router
        route = [nodes.pop(nodes.index(current_router))]
        while len(nodes):
            # 2. randomly choose, with equal probability, one of the nodes that
            # is SURELY connected to the current node to be the next in path
            next_router = random.choice(nodes)

            # SURELY: check whether there is an edge/link/connection or not
            if adj_mtx[current_router][next_router]: 
                # 3. if the chosen node hasn't been visited before, mark it as the
                # next in the path (gene). Otherwise find another node
                current_router = next_router
                route.append(nodes.pop(nodes.index(current_router)))

                # 6. do this until the destination node is found
                if current_router == end_router:
                    break

                count = 0
            else:
                # max trials to find a valid path: average of 100 chances per gene
                count += 1
                if count > 100:
                    route = False
                    break

        if route and len(route) > num_nodes:
            route = False

        return route
    

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
                
        #attractiveness
        attractiveness_beta_at_distance_0 = 1
        distance_of_two_fireflies = (l+1)/(wl_avail+0.0000001)
        attractiveness_beta = attractiveness_beta_at_distance_0 * math.exp(-self.absorption_coefficient_gamma * distance_of_two_fireflies**2)

        return L, wl_avail, l+1, attractiveness_beta  # Label, λ's available, length of route and attractiveness
    
    

    # TODO: Q: Can I select the same father/mother twice?
    # TODO: Selection Function
    # [[chrom], [L], wl_avail, r_len]
    def select(self, population, Tc, times=3):
        """ Tournament """
        parents = []
        while len(population)-1:
            # choose and pop a random candidate from population
            # it pops because it cannot be selected twice
            candidate = [random.choice(population)]
            if Tc > random.random():
                for tourn_times in range(times):
                    # choose another candidate and compare two fitnesses
                    # the winner becomes the top candidate; loser is eliminated
                    candidate.append(random.choice(population))
                    if candidate[0][2] >= candidate[1][2]:
                        candidate.remove(candidate[1])
                    else: # fit[1] > fit[0] .:. candidate[0] eliminated
                        candidate.remove(candidate[0])
                parents.append(candidate[0][0])
            population.remove(candidate[0]) # A: no! never the same dad/mum

        return parents

    # TODO: Crossover Function
    def cross(self, parents):
        """ One Point """
        children = []
        while len(parents)-1 > 0:
            # choose parents and make sure they are differente ones
            dad = random.choice(parents)
            mom = random.choice(parents)
            parents.remove(dad)
            
            # avoid crossing twins: check if parents are the same individual
            for i in range(10):
                if dad == mom:
                    mom = random.choice(parents)
                else:
                    parents.remove(mom)
                    break
    
            # common nodes between father and mother, excluding source and target
            index_routers = []
            for gene in dad[1:len(dad)-1]:
                if gene in mom[1:len(mom)-1]:
                    index_routers.append([dad.index(gene), mom.index(gene)])
    
                if len(index_routers):
                    # randomly choose a common node index to be the crossover point
                    common_router = random.choice(index_routers)
                    children.append(dad[:common_router[0]] + mom[common_router[1]:])
                    children.append(mom[:common_router[1]] + dad[common_router[0]:])

            if not len(children):
                children = False
                
            if children:
                for child in children:
                    child = self.remove_duplicate(child)

            return children

    # TODO: Mutation Function
    def mutate(self, adj_mtx, normal_chrom):
        # DO NOT perform mutation if:
        # route has only one link which directly connects source to target
        if len(normal_chrom) == 2:
            return normal_chrom

        trans_chrom = list(normal_chrom) # CAN'T NORMALLY COPY THIS

        # choose a random mutation point, excluding the first and the last
        geneid = random.randrange(1, len(normal_chrom)-1)

        # extract or pop() source and target nodes from chromosome
        start_router = trans_chrom.pop(geneid)
        end_router   = trans_chrom.pop()

        # remove all genes after mutation point
        for gene in range(geneid, len(trans_chrom)):
            trans_chrom.pop()

        # alphabet: graph vertices that are not in genes before mutation point
        allels = [start_router, end_router]
        allels += [a for a in range(14) if a not in trans_chrom]

        # create a new route R from mutation point to target node
        R = self.make_chromosome(adj_mtx, start_router, end_router, allels)

        # check if new route/path is valid
        if R:
            trans_chrom += R
        else:
            trans_chrom = list(normal_chrom)

        return trans_chrom
    
    
    def remove_duplicate(self, route):
        for i in range(len(route)):
            for j in range(1, len(route)-i):
                if(route[-j]==route[i]):
                    route = route[:i]+route[-j:]
                    break
        return route

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