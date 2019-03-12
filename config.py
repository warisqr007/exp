
import os

# Debug Parameters
DEBUG = False

RESULT_DIR = os.path.join(os.path.abspath(__file__), 'results')

# Simulation Parameters
SIM_NUM_CALLS = 200
SIM_MIN_LOAD  = 1
SIM_MAX_LOAD  = 31

SIM_LOAD = 120

# NET Parameters
NET_NUM_CHANNELS  = 16   # total number of wavelengths available
NET_CHANNEL_FREE  = False  # init all link wavelengths available at once?
NET_CHANNEL_BIAS  = 0.50   # probability of free λ vs occupied λ, respect.

NET_NSF    = True  # use NSF network topology in simulation?
NET_ARPA   = True  # use ARPA    network topology in simulation?
NET_CLARA  = True  # use CLARA   network topology in simulation?
NET_ITA    = True  # use ITALIAN network topology in simulation?
NET_JANET  = True  # use JANET   network topology in simulation?
NET_RNP    = True  # use RNP     network topology in simulation?

ALG_DFF    = True  # use 'Dijkstra + First Fit' algoritihms in simulation?

ALG_DFF_G    = True

ALG_GA     = True  # use 'Genetic Algorithm'                      in simulation?  

ALG_GA_G     = True

ALG_FFGA   = True  # use 'FireFly-Genetic Algorithm hybrid'

ALG_FFGA_G   = True

ALG_FF     = True  #  use 'Firefly Algorithm'

ALG_FF_G     = True

#Firefly parameters

FF_NUM            = 20
FF_MIN_GEN        = 25     # min number of generations
FF_MAX_GEN        = 80     # max number of generations

FF_MIN_CROSS_RATE = 0.15   # min crossover rate
FF_MAX_CROSS_RATE = 0.40   # max crossover rate

absorption_coefficient_gamma = 1.0


# TODO: pass everything as a dict to the class
# GA Parameters
GA_SIZE_POP       = 30     # size of the population of each species

GA_MIN_GEN        = 25     # min number of generations
GA_MAX_GEN        = 80     # max number of generations

GA_MIN_CROSS_RATE = 0.15   # min crossover rate
GA_MAX_CROSS_RATE = 0.40   # max crossover rate

GA_MIN_MUT_RATE   = 0.02   # min mutation rate
GA_MAX_MUT_RATE   = 0.20   # max mutation rate

GA_GEN_INTERVAL   = 8      # interval to update rates



### EOF ###


