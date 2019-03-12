import random
import numpy as np

class Network(object):
    """ Network: superclass """

    __wavelength_matrix = None   # W (3D numpy array/matrix)
    __adjacency_matrix  = None   # A (2D numpy array/matrix)
    __traffic_matrix    = None   # T (dict: 3D np array + dict of dicts)
    __connection_matrix = None  # C (dict of dicts)

    def __init__(self, ch_num, ch_free, ch_bias):
        self.num_channels      = ch_num  # int
        self.channel_init_free = ch_free # bool
        self.channel_init_bias = ch_bias # probs for a given λ be *NOT* free 
        self.allow_multi_od    = True    # allow multiple OD conn. pairs?

        self.wave_mtx     = None  # W copy
        self.adj_mtx      = None  # A copy
        self.traffic_mtx  = None  # T copy
        self.conn_mtx     = None  # C copy

    def init_network(self):
        """ Generate matrices W, A and T """

        # define links or edges as node index pairs 
        links = self.get_edges()

        dimension = (self.num_nodes, self.num_nodes)
        a_mtx = np.zeros(dimension, dtype=np.uint8)
        for l in links:
            a_mtx[l[0]][l[1]] = 1
            a_mtx[l[1]][l[0]] = a_mtx[l[0]][l[1]] # symmetric

        dimension = (self.num_nodes, self.num_nodes, self.num_channels)
        time_mtx = np.zeros(dimension, dtype=np.float64)
        t_mtx = {}
        t_mtx['time'] = time_mtx
        t_mtx['num_calls'] = 0
        t_mtx['wave_usg'] = np.zeros(self.num_channels, dtype=np.uint8) 
        
        c_mtx = {}
       
        dimension = (self.num_nodes, self.num_nodes, self.num_channels)
        w_mtx = np.zeros(dimension, dtype=np.float64)
        for i, j in links:
            for w in range(self.num_channels):
                w_mtx[i][j][w] = 1
                w_mtx[j][i][w] = 1 # symmetric
        for conn in t_mtx:
            if isinstance(conn, tuple):
                w = conn[2]
                for path in t_mtx[conn]:
                    for i in range(len(path)-1):
                        rcurr = path[i]
                        rnext = path[i+1]
                        w_mtx[rcurr][rnext][w] = 0
                        w_mtx[rnext][rcurr][w] = 0 # symmetric

        # finally, init constants
        self.__wavelength_matrix = w_mtx
        self.__adjacency_matrix  = a_mtx
        self.__traffic_matrix    = t_mtx
        self.__connection_matrix = c_mtx

    def reset_network(self):
        """ reset all networks the their previous initial stages """

        self.wave_mtx    = self.__wavelength_matrix.copy() # matrix
        self.adj_mtx     = self.__adjacency_matrix.copy()  # matrix
        self.traffic_mtx = self.__traffic_matrix.copy()    # dict
        self.conn_mtx    = self.__connection_matrix.copy()

    # on traffic matrix, update all channels that are still being used
    def update_network(self, until_next):
        
        for key in list(self.traffic_mtx): 
            if not isinstance(key, tuple): # ensure we don't mess with the 3D mtx
                continue # because we want @key to be (ODλ)
            w = key[2] # extract λ in question
            for path in list(self.traffic_mtx[key]):
                if self.traffic_mtx[key][path] > until_next:
                    # update both dict value and time matrix values
                    self.traffic_mtx[key][path] -= until_next
                    for i in range(len(path)-1):
                        rcurr = path[i]
                        rnext = path[i+1]
                        self.traffic_mtx['time'][rcurr][rnext][w] -= until_next
                        self.traffic_mtx['time'][rnext][rcurr][w] -= until_next
                        
                        for n in range(i+2,len(path)):
                            key_c = (rcurr, path[n-1], w)
                            path_c = path[i:n]
                            self.conn_mtx[key_c][path_c] -= until_next
                        
                else:
                    # remove lightpath from traffic matrix (pop) and update both
                    # time and wavelength availability matrices
                    self.traffic_mtx[key].pop(path)
                    for i in range(len(path)-1):
                        rcurr = path[i]
                        rnext = path[i+1]
                        # the time until the next call is now 0, since the
                        # connection that was withholding λ it is gone
                        self.traffic_mtx['time'][rcurr][rnext][w] = 0.0
                        self.traffic_mtx['time'][rnext][rcurr][w] = 0.0
                        # do not forget to free the respective λ as well
                        self.wave_mtx[rcurr][rnext][w] = 1.0
                        self.wave_mtx[rnext][rcurr][w] = 1.0
                        
                        for n in range(i+2,len(path)):
                            key_c = (rcurr, path[n-1], w)
                            path_c = path[i:n]
                            self.conn_mtx[key_c].pop(path_c)
                        
                        

    def is_same_traffic_entry(self, odw_key, dow_key):
        """ compare two dict keys from traffic matrix in order to checj whether
        they are equal not
        """

        if not isinstance(od_key, tuple) or not isinstance(do_key, tuple):
            sys.stderr.write('something is wrong\n')
            return False
        elif len(od_key) != 3 or len(do_key) != 3:
            sys.stderr.write('something is wrong\n')
            return False

        # check direct equality
        if odw_key[0] == dow_key[0] and odw_key[1] == dow_key[1]:
            if odw_key[2] == dow_key[2]:
                return True
            else:
                return False
        # check reversed equality
        elif odw_key[0] == dow_key[1] and odw_key[1] == dow_key[0]:
            if odw_key[2] == dow_key[2]:
                return True
            else:
                return False
        else:
            return False


class NationalScienceFoundation(Network): 
    """ U.S. National Science Foundation Network (NSFNET) """

    def __init__(self, ch_n, ch_f, ch_b):
        super(NationalScienceFoundation, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'NSF'
        self.fullname    = 'National Science Foundation'
        self.num_nodes   = len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 12  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [(0,1), (0,2), (0,5),   #  0
            (1,2), (1,3),          #  1
            (2,8),                 #  2
            (3,4), (3,6), (3,13),  #  3
            (4,9),                 #  4
            (5,6), (5,10),         #  5
            (6,7),                 #  6
            (7,8),                 #  7
            (8,9),                 #  8
            (9,11), (9,12),        #  9
            (10,11), (10,12),      # 10
            (11,13)                # 11
        ]

    def get_nodes_2D_pos(self):
        """ Get position of the nodes on the bidimensional Cartesian plan

            This might be useful for plotting the topology as a undirected, 
            unweighted graph
        """
        return [['0',  (0.70, 2.70)], #  0
            ['1',  (1.20, 1.70)], #  1
            ['2',  (1.00, 4.00)], #  2
            ['3',  (3.10, 1.00)], #  3
            ['4',  (4.90, 0.70)], #  4
            ['5',  (2.00, 2.74)], #  5
            ['6',  (2.90, 2.66)], #  6
            ['7',  (3.70, 2.80)], #  7
            ['8',  (4.60, 2.80)], #  8
            ['9',  (5.80, 3.10)], #  9
            ['10', (5.50, 3.90)], # 10
            ['11', (6.60, 4.60)], # 11
            ['12', (7.40, 3.30)], # 12
            ['13', (6.50, 2.40)]  # 13
        ]

class AdvancedResearchProjectsAgency(Network): 
    """ U.S. Advanced Research Projects Agency (ARPANET) """
    def __init__(self, ch_n, ch_f, ch_b):
        super(AdvancedResearchProjectsAgency, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'ARPA'
        self.fullname    = 'Advanced Research Projects Agency'
        self.num_nodes   = len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False FIXME
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 12  # destination node defined for all connections

    def get_edges(self):
        """ get edges """

        return [\
            (0,1), (0,2), (0,19),  #  0
            (1,2), (1,3),          #  1
            (2,4),                 #  2
            (3,4), (3,5),          #  3
            (4,6),                 #  4
            (5,6), (5,7),          #  5
            (6,9),                 #  6
            (7,8), (7,9), (7,10),  #  7
            (8,9), (8,19),         #  8
            (9,15),                #  9
            (10,11), (10,12),      # 10
            (11,12),               # 11
            (12,13),               # 12
            (13,14), (13,16),      # 13
            (14,15),               # 14
            (15,17), (15,18),      # 15
            (16,17), (16,19),      # 16
            (17,18)                # 17
        ]

    def get_nodes_2D_pos(self):
        """ Get position of the nodes on the bidimensional Cartesian plan
    
            This might be useful for plotting the topology as a undirected, 
            unweighted graph
        """
        return [\
            ['0',  (1.80, 5.70)], #  0
            ['1',  (2.80, 5.00)], #  1
            ['2',  (3.40, 6.30)], #  2
            ['3',  (3.40, 5.50)], #  3
            ['4',  (4.50, 5.60)], #  4
            ['5',  (4.70, 4.60)], #  5
            ['6',  (5.30, 4.80)], #  6
            ['7',  (3.60, 4.40)], #  7
            ['8',  (2.20, 4.00)], #  8
            ['9',  (4.80, 3.50)], #  9
            ['10', (2.40, 2.60)], # 10
            ['11', (2.50, 1.50)], # 11
            ['12', (1.40, 2.30)], # 12
            ['13', (1.80, 3.20)], # 13
            ['14', (3.70, 2.70)], # 14
            ['15', (5.20, 2.50)], # 15
            ['16', (0.80, 3.90)], # 16
            ['17', (1.20, 0.50)], # 17
            ['18', (3.60, 0.80)], # 18
            ['19', (0.80, 5.50)]  # 19
        ]

class CooperacionLatinoAmericana(Network): 
    """ Cooperación Latino Americana de Redes Avanzadas (RedClara) """

    def __init__(self, ch_n, ch_f, ch_b):
        super(CooperacionLatinoAmericana, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'CLARA'
        self.fullname    = u'Cooperación Latino Americana de Redes Avanzadas'
        self.num_nodes   = len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False FIXME
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 12  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [\
            (0,1), (0,5), (0,8), (0,11),  #  0
            (1,2),                        #  1
            (2,3),                        #  2
            (3,4),                        #  3
            (4,5),                        #  4
            (5,6), (5,7), (5,11),         #  5
            (7,8),                        #  7
            (8,9), (8,11),                #  8
            (9,10), (9,11),               #  9
            (11,12)                       # 11
        ]

    def get_nodes_2D_pos(self):
        """ Get position of the nodes on the bidimensional Cartesian plan
    
            This might be useful for plotting the topology as a undirected, 
            unweighted graph
        """
        return [\
            ['US', (2.00, 6.00)], #  0
            ['MX', (1.00, 6.00)], #  1
            ['GT', (1.00, 4.50)], #  2
            ['SV', (1.00, 2.50)], #  3
            ['CR', (1.00, 1.00)], #  4
            ['PN', (2.00, 1.00)], #  5
            ['VE', (1.50, 1.70)], #  6
            ['CO', (3.00, 1.00)], #  7
            ['CL', (4.00, 1.00)], #  8
            ['AR', (5.00, 3.50)], #  9
            ['UY', (5.00, 1.00)], # 10
            ['BR', (4.00, 6.00)], # 11
            ['UK', (5.00, 6.00)]  # 12
        ]

class Italian(Network): 
    """ Italian Network (NSFNET) """

    def __init__(self, ch_n, ch_f, ch_b):
        super(Italian, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'ITA'
        self.fullname    = u'Italian'
        self.num_nodes   = len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 12  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [\
            (0,1), (0,2),                        #  0
            (1,2), (1,3), (1,4),                 #  1
            (2,7), (2,8), (2,9),                 #  2
            (3,4), (3,5),                        #  3
            (4,6), (4,7),                        #  4
            (5,6),                               #  5
            (6,7),                               #  6
            (7,9), (7,10),                       #  7
            (8,9), (8,12),                       #  8
            (9,11), (9,12),                      #  9
            (10,13),                             # 10
            (11,12), (11,13),                    # 11
            (12,14), (12,20),                    # 12
            (13,14), (13,15),                    # 13
            (14,15), (14,16), (14,18), (14,19),  # 14
            (15,16),                             # 15
            (16,17),                             # 16
            (17,18),                             # 17
            (18,19),                             # 18
            (19,20)                              # 19
        ]

    def get_nodes_2D_pos(self):
        """ Get position of the nodes on the bidimensional Cartesian plan

            This might be useful for plotting the topology as a undirected, 
            unweighted graph
        """
        return [\
            ['x', (0.70, 6.50)], #  0 
            ['x', (1.80, 7.00)], #  1
            ['x', (1.80, 6.00)], #  2
            ['x', (3.00, 7.70)], #  3
            ['x', (2.70, 6.80)], #  4
            ['x', (4.00, 6.70)], #  5
            ['x', (3.30, 6.30)], #  6
            ['x', (2.90, 5.70)], #  7
            ['x', (2.00, 5.00)], #  8
            ['x', (2.90, 5.00)], #  9
            ['x', (3.80, 5.20)], # 10
            ['x', (3.20, 4.50)], # 11
            ['x', (2.50, 3.50)], # 12
            ['x', (3.90, 4.00)], # 13
            ['x', (3.70, 2.50)], # 14
            ['x', (4.90, 3.00)], # 15
            ['x', (4.50, 2.00)], # 16
            ['x', (4.70, 1.00)], # 17
            ['x', (3.80, 0.50)], # 18
            ['x', (2.70, 0.60)], # 19
            ['x', (1.20, 1.50)]  # 20
        ]

class JointAcademic(Network): 
    """ U.K. Joint Academic Network (JANET) """

    def __init__(self, ch_n, ch_f, ch_b):
        super(JointAcademic, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'JANET'
        self.fullname    = 'Joint Academic Network'
        self.num_nodes   = len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # FIXME the below are used iff multi_SD is set to False
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 12  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [\
            (0,1), (0,2),                # 0
            (1,2), (1,3),                # 1
            (2,4),                       # 2
            (3,4), (3,5), #(3,6),        # 3
            (4,6),                       # 4
            (5,6)                        # 5
        ]

    def get_nodes_2D_pos(self):
        """ Get position of the nodes on the bidimensional Cartesian plan
            This might be useful for plotting the topology as a undirected, 
            unweighted graph
        """
        return [\
            ['Gla', (1.50, 4.00)], # 0
            ['Man', (1.00, 3.00)], # 1
            ['Lee', (2.00, 3.00)], # 2
            ['Bir', (1.00, 2.00)], # 3
            ['Not', (2.00, 2.00)], # 4
            ['Bri', (1.00, 1.00)], # 5
            ['Lon', (2.00, 1.00)]  # 6
        ]


class RedeNacionalPesquisa(Network): 
    """ Rede (Brasileira) Nacional de Pesquisa (Rede Ipê / RNP) """

    def __init__(self, ch_n, ch_f, ch_b):
        super(RedeNacionalPesquisa, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'RNP'
        self.fullname    = u'Rede Nacional de Pesquisas (Rede Ipê)'
        self.num_nodes   = len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 12  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [\
            (0,1),                                #  0
            (1,3), (1,4),                         #  1
            (2,4),                                #  2
            (3,4), (3,7), (3,17), (3,19), (3,25), #  3
            (4,6), (4,12),                        #  4
            (5,25),                               #  5
            (6,7),                                #  6
            (7,8), (7,11), (7,18), (7,19),        #  7
            (8,9),                                #  8
            (9,10),                               #  9
            (10,11),                              # 10
            (11,12), (11,13), (11,15),            # 11
            (13,14),                              # 13
            (14,15),                              # 14
            (15,16), (15,19),                     # 15
            (16,17),                              # 16
            (17,18),                              # 17
            (18,19), (18,20), (18,22),            # 18
            (20,21),                              # 20
            (21,22),                              # 21
            (22,23),                              # 22
            (23,24),                              # 23
            (24,25), (24,26),                     # 24
            (26,27)                               # 26
    ]

    def get_nodes_2D_pos(self):
        """ Get position of the nodes on the bidimensional Cartesian plan
    
            This might be useful for plotting the topology as a undirected, 
            unweighted graph
        """
        return [\
            ['RR', (5.00,  3.25)], #  0
            ['AM', (5.50,  3.75)], #  1
            ['AP', (8.25,  3.75)], #  2
            ['DF', (4.00,  5.00)], #  3
            ['PA', (9.00,  3.00)], #  4
            ['TO', (3.00,  3.00)], #  5
            ['MA', (9.00,  4.00)], #  6
            ['CE', (9.50,  5.00)], #  7
            ['RN', (10.50, 5.00)], #  8
            ['PB', (10.50, 3.00)], #  9
            ['PB', (10.50, 1.00)], # 10
            ['PE', (9.50,  1.00)], # 11
            ['PI', (9.00,  2.00)], # 12
            ['AL', (8.00,  2.00)], # 13
            ['SE', (7.00,  2.00)], # 14
            ['BA', (6.00,  2.00)], # 15
            ['ES', (6.00,  1.00)], # 16
            ['RJ', (4.00,  1.00)], # 17
            ['SP', (2.00,  1.00)], # 18
            ['MG', (6.00,  5.50)], # 19
            ['SC', (1.00,  1.00)], # 20
            ['RS', (1.00,  2.00)], # 21
            ['PR', (2.00,  2.00)], # 22
            ['MS', (2.00,  4.00)], # 23
            ['MT', (2.00,  5.00)], # 24
            ['GO', (3.00,  5.00)], # 25
            ['RO', (1.00,  5.00)], # 26
            ['AC', (1.00,  4.00)]  # 27
        ]
    
    
class TestingNetworkLinear(Network): 
    def __init__(self, ch_n, ch_f, ch_b):
        super(TestingNetworkLinear, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'TNL'
        self.fullname    = 'Testing Network Linear'
        self.num_nodes   = 6 #len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 5  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [(0,1),   #  0
                (1,2),   #  1
                (2,3),   #  2
                (3,4),   #  3
                (4,5)    #  4
               ]

    
class TestingNetworkRing(Network):

    def __init__(self, ch_n, ch_f, ch_b):
        super(TestingNetworkRing, self).__init__(ch_n, ch_f, ch_b)
        self.name        = 'TNR'
        self.fullname    = 'Testing Network Ring'
        self.num_nodes   = 6 #len(self.get_nodes_2D_pos())
        self.num_links   = len(self.get_edges())

        # the below are used iff multi_SD is set to False
        self.source_node = 0   # source node defined for all connections
        self.dest_node   = 5  # destination node defined for all connections

    def get_edges(self):
        """ get """
        return [(0,1),   #  0
                (1,2),   #  1
                (2,3),   #  2
                (3,4),   #  3
                (4,5),   #  4
                (5,0)    #  5
               ]

### EOF ###


