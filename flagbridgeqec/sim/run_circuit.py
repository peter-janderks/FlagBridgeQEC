import sparse_pauli as sp
from flagbridgeqec.circuits import cir_steane_5a
from check_ft1_parallel import Check_FT
#import check_ft1_parallel as cf
from flagbridgeqec.utils import error_model_2 as em
import circuit_metric as cm
from operator import mul
from functools import reduce
import numpy as np
#from circuits.a4_circuit import *
#from a4_circuit import *
#from a5_circuit import *
#from a6_circuit import *

product = lambda itrbl: reduce(mul, itrbl)

LOCS = dict()
LOCS['SINGLE_GATES'] = ['S', 'H',
                        'X90', 'Y90', 'Z90',
                        'X', 'Y', 'Z',
                        'X180', 'Y180', 'Z180']
LOCS['DOUBLE_GATES'] = ['CNOT', 'CPHASE', 'ZZ90', 'SWAP']
LOCS['PREPARATIONS'] = ['P_X', 'P_Z', 'P']
LOCS['MEASUREMENTS'] = ['M_X', 'M_Z', 'M']
LOCS['idle'] = ['I']

class FT_protocol(object):
    def __init__(self, per, cir_id=10, p_idling=1):
        self.p1 = per
        self.pI = p_idling*per
        self.esm_circuits = self.esm_circuits_steane(cir_id)

        if cir_id == '6a':
            # for Steane-cl-L2                     
            self.q_synd = [8,10,12,80,100,120]
            self.q_flag = [130, 110, 90, 9, 11, 13]
            self.q_syndx = [8, 10, 12]
            self.q_syndz = [80, 100, 120]
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))
        elif cir_id == '4a':
            
            self.q_synd = [8,9,10,80,90,100]
            self.q_flag = [11, 110]
            self.q_syndx = [8, 9, 10]
            self.q_syndz = [80, 90, 100]
            self.x_ancillas = set((80,90,100,110))
            self.z_ancillas = set((8,9,10,11))
        elif cir_id == '5a':

            self.q_synd = [8,10,12,80,100,120]
            self.q_flag = [9,11,90,110]
            self.q_syndx = [80,100,120]
            self.q_syndz = [8,10,12]
            self.x_ancillas = set((80,90,100,110,120))
            self.z_ancillas = set((8,9,10,11,12))

        self.cir_index = cir_id

        self.ancillas = self.q_synd + self.q_flag

        self.lut_synd, self.lut_flag = Check_FT(cir_id).lut_gen()
        
        self.stabilisers = {'X': {9: sp.Pauli(x_set=[1,2,4,5]), 
                                  11: sp.Pauli(x_set=[1,3,4,7]),
                                  13: sp.Pauli(x_set=[4,5,6,7])},
                            'Z': {80: sp.Pauli(z_set=[1,2,4,5]), 
                                  100: sp.Pauli(z_set=[1,3,4,7]), 
                                  120: sp.Pauli(z_set=[4,5,6,7])}}
        self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}
        self.init_logs = [sp.Pauli(x_set=[1,2,3,4,5,6,7]), sp.Pauli(z_set=[1,2,3,4,5,6,7])]

    def esm_circuits_steane(self,cir_id):
        if cir_id == '6a':
            esm_circuits = []
            esm_circuits.extend([cir_steane_6a(1),cir_steane_6a(2),cir_steane_6a(3),cir_steane_6a(4)])
        elif cir_id == '4a':
            esm_circuits = []
            esm_circuits.extend([[cir_steane_4a(1),cir_steane_4a(2)],cir_steane_4a(3), cir_steane_4a(4), cir_steane_4a(5)])
        elif cir_id == '5a':
            esm_circuits = []
            esm_circuits.extend([cir_steane_5a(1),cir_steane_5a(2),cir_steane_5a(3),cir_steane_5a(4)])

        return(esm_circuits)


    def run(self, trials = 1):
        total_errors = 0 
        for trial in range(trials):
            err = sp.Pauli()
            corr = sp.Pauli()
            synd1 = set()
            synd2 = set()

            synd_err = circ2err(self.esm_circuits[0], fowler_model(self.esm_circuits[0], 
                                                                   self.p1, self.p1, 
                                                                   self.p1, self.pI), 
                                err)
            err.prep(self.z_ancillas)
            synd1 |= synd_err[0]
            err = synd_err[1]
                            
            if len(synd1):
                subcir = self.esm_circuits[2]
                synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p1, 
                                                         self.p1, self.pI), err)
                err.prep(self.ancillas)
                synd2 |= synd_err[0]
                err = synd_err[1]
                flag_qs = synd1 & set(self.q_flag)
                synd_qx = sorted(synd2 & set(self.q_syndx))
                synd_qz = sorted(synd2 & set(self.q_syndz))

                if len(flag_qs):
                    try:
                        corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qx)]
                        corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qz)]
        
                    except KeyError:
                        # Key is not present
                        print(flag_qs)
                        print('not in lut')
                else:
                    try:
                        corr *= self.lut_synd[tuple(synd_qz)]
                        corr *= self.lut_synd[tuple(synd_qx)]
                    except KeyError:
                        print('not in lut')
            else:
                subcir = self.esm_circuits[1]
                synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p1, self.p1, self.pI), err)
                err.prep(self.x_ancillas)
                synd1 |= synd_err[0]
                err = synd_err[1]
                if len(synd1):

                    subcir = self.esm_circuits[3]
                    synd_err = circ2err(subcir, fowler_model(subcir, 
                                                                     self.p1, self.p1, self.p1, 
                                                                     self.pI), err)
                    err.prep(self.ancillas)
                    synd2 |= synd_err[0]
                    err = synd_err[1]
                    flag_qs = synd1 & set(self.q_flag)
                    synd_qx = sorted(synd2 & set(self.q_syndx))
                    synd_qz = sorted(synd2 & set(self.q_syndz))
                    if len(flag_qs):
                        try:
                            corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qx)]
                            corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qz)]
                        except KeyError:
                            print('not in lut')
                    else:
                        try:
                            corr *= self.lut_synd[tuple(synd_qz)]
                            corr *= self.lut_synd[tuple(synd_qx)]
                        except KeyError:
                            print('not in lut')

            err_fnl = err
            err *= corr
            error_before_p_r = err
            err_tp = singlelogical_error(err, self.init_logs)

            # run another perfect round to clean the left errors                 
            synd_fnl = set()
            for i in range(0,2):
                subcir = self.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, 0, 0, 0), err)
                synd_fnl |= synd_err[0]
                err = synd_err[1]

            err.prep(self.ancillas)
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndx)))]
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndz)))]
            # # check logical errors                                             
            err_tp = singlelogical_error(err, self.init_logs)
            if err_tp != 'I':
                print(err_tp)
                total_errors +=1
        print(total_errors)
        return(total_errors)

def circ2err(circ, err_model, err):
    synd = set()
    for stp, mdl in list(zip(circ, err_model)):
        # run timestep, then sample             
        new_synds, err = cm.apply_step(stp, err)
        err *= product(_.sample() for _ in mdl)
        synd |= new_synds[1]
    new_synds, err = cm.apply_step(circ[-1], err)
    synd |= new_synds[1]
    return synd, err

def fowler_model(extractor, p1, p2, pm, pI=0):
    """                                                                                        
    Produces a circuit-level error model for a given syndrome extractor                        
    """

    err_list = [[] for _ in extractor]
    for t, timestep in enumerate(extractor):

        singles, doubles = [[tp[1:] for tp in timestep if tp[0] in LOCS[_]]
                            for _ in ['SINGLE_GATES', 'DOUBLE_GATES']]

        p, m = [[tp[1:] for tp in timestep if tp[0] == s]
                                 for s in ('P', 'M')]

        idles = [tp[1:] for tp in timestep if tp[0] == 'I']

        err_list[t].extend([
            em.depolarize(p1, singles),
            em.depolarize(pI, idles),
            em.pair_twirl(p2, doubles),
            em.x_flip(pm, p)
            ])

        err_list[t - 1].extend([
            em.x_flip(pm, m)
                    ])

    return err_list

def singlelogical_error(err_fnl, corr_logs):
    """
    returns a single letter recording the resulting logical error (may be I,X, Y or Z)
    """

    x_log = corr_logs[0]
    z_log = corr_logs[1]

    anticom_dict = {
        (0, 0): 'I',
        (0, 1): 'X',
        (1, 0): 'Z',
        (1, 1): 'Y'
    }
    x_com, z_com = x_log.com(err_fnl), z_log.com(err_fnl)
    return(anticom_dict[ ( x_com, z_com ) ])

def check(per,trials, cir_id,p_idling):
    print(trials,'trials')
    x = FT_protocol(per,cir_id,p_idling)
    err = x.run(trials)
    return(err)

if __name__ == '__main__':
    x = FT_protocol(0.001,cir_id='5a',p_idling=1)
    error = x.run(1000)
