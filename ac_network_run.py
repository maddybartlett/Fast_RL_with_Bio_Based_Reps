import sys
sys.path.insert(0, './network')

import matplotlib.pyplot as plt
import numpy as np
import os 

import learnrules as rules
import representations as rp
from ac_learn import ActorCriticLearn

def main(args):
    REPS = {'1H' : rp.OneHotRep((8,8,4)), 'SSP': rp.SSPRep(N=3, D=args['dims'], scale=[0.75,0.75,1.0]), 'Grid' : rp.GridSSPRep(3)}
    RULES = {'TD0': rules.ActorCriticTD0, 'TDLambda': rules.ActorCriticTDLambda}
    
    ep_results = []
    ep_rewards=[]
    ep_values=[]
    
    out = ActorCriticLearn().run(rep = REPS[args['rep']], 
                                 rule = RULES[args['rule']],
                                 alpha = args['alpha'], 
                                 beta = args['beta'], 
                                 gamma = args['gamma'], 
                                 lambd = args['lambda'],
                                 n_neurons = args['neurons'], 
                                 sparsity = args['sparsity'], 
                                 dims = args['dims'],
                                 verbose = False, 
                                 trials = 500,
                                data_dir = args['data_dir'])

    ep_results.append(out["episodes"])
    ep_rewards.append(out["rewards"])
    ep_values.append(out["values"])
    
    ep_rews = []
    
    for i in range(len(ep_rewards[0])):
        ep_rews.append(np.sum(ep_rewards[0][i]))
    
    plt.figure(figsize = (15,5))
    plt.plot(ep_rews)
    plt.ylabel('Reward')
    plt.xlabel('Learning Trial')
    plt.show()
    plt.savefig(args['data_dir']+"/ep_rewards.pdf")
                                       
if __name__ == '__main__':
    directory = input("Enter path for saving the data? ") 
    while not os.path.exists(directory):
        print("Path of the file is Invalid")
        directory = input("Enter path for saving the data? ")
        
    representation = "SSP"        
    rule = "TDLambda"
        
    alpha = 0.64539
    beta = 0.655092
    gamma = 0.832537
    lambd = 0.828999
    
    neurons = 4427
    sparsity = 0.552192
    dims = 256
    
    params = {'rep':representation, 'rule': rule, 'alpha':alpha,
             'beta': beta, 'gamma':gamma, 'lambda': lambd, 'neurons':neurons,
             'sparsity':sparsity, 'dims':dims, 'data_dir':directory}
    main(params)
