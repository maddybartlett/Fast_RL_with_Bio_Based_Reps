import sys
sys.path.insert(0, './network')

import learnrules as rules
import representations as rp
from ac_learn import ActorCriticLearn
import nni
import numpy as np

def main(args):
    out = ActorCriticLearn().run(env = 'MiniGrid', rep = rp.OneHotRep((8,8,4)), rule = rules.ActorCriticTDLambda,
                                 alpha = args['alpha'], beta = args['beta'], gamma = args['gamma'], lambd = args['lambda'],
                                 n_neurons = args['neurons'], sparsity = args['sparsity'],
                                verbose = False, trials = 2000, data_dir = 'data\main_experiment\MG_TDLambda_1H_long')
    result=[index for index,val in enumerate(out['roll_mean'][0]) if val > 0.95]
    nni.report_final_result(result[0])
                                       
if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)