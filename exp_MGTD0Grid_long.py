import sys
sys.path.insert(0, './network')

import learnrules as rules
import representations as rp
from ac_learn import ActorCriticLearn
import nni

def main(args):
    out = ActorCriticLearn().run(rep = rp.GridSSPRep(3), alpha = args['alpha'], 
                                 beta = args['beta'], gamma = args['gamma'], n_neurons = args['neurons'], 
                                 sparsity = args['sparsity'], verbose = False, trials = 2000,
                                data_dir = 'data\main_experiment\MG_TD0_Grid_long')
    result=[index for index,val in enumerate(out['roll_mean'][0]) if val > 0.95]
    nni.report_final_result(result[0])
                                       
if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)