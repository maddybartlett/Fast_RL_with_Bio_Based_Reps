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
    representation = input("1H, SSP or Grid representation? ")
    while not (representation == '1H' or representation == 'SSP' or representation == 'Grid'):
        print('ValueError: Value entered is not a valid representation method. Must be either 1H, SSP or Grid.')
        representation = input("1H, SSP or Grid representation? ")
        
    rule = input("TD0 or TDLambda? ")
    while not (rule == 'TD0' or rule == 'TDLambda'):
        print('ValueError: Value entered is not a valid rule. Must be either TD0 or TDLambda.')
        rule = input("TD0 or TDLambda? ")
        
    alpha = float(input("Enter value for learning rate: "))
    beta = float(input("Enter value for action value discount: "))
    gamma = float(input("Enter value for state value discount: "))
    
    neurons = int(input("Enter number of neurons: "))
    sparsity = float(input("Enter value for sparsity: "))
    
    directory = input("Enter path for saving the data? ") 
    while not os.path.exists(directory):
        print("Path of the file is Invalid")
        directory = input("Enter path for saving the data? ")
    
    if rule == 'TDLambda':
        lambd = float(input("Enter value for eligibility trace discount (lambda): "))
    else:
        lambd = None
    
    if representation == 'SSP':
        dims = int(input("Enter value for dimensionality of SSP representation: "))
    else:
        dims = 1
    
    params = {'rep':representation, 'rule': rule, 'alpha':alpha,
             'beta': beta, 'gamma':gamma, 'lambda': lambd, 'neurons':neurons,
             'sparsity':sparsity, 'dims':dims, 'data_dir':directory}
    main(params)
