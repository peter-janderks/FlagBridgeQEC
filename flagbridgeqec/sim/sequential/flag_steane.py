import sparse_pauli as sp
from flagbridgeqec.circuits import esm2, esm3, esm4, esm5, esm7
from flagbridgeqec.sim.sequential.check_ft1 import Check_FT
import flagbridgeqec.sim.sequential.check_ft1 as cf
from flagbridgeqec.utils import error_model_2 as em
import circuit_metric as cm
from operator import mul
from functools import reduce
import numpy as np


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


class Steane_FT(object):
    def __init__(self, p1, p2, pm, cir_id=2, idling=False, ridle=0):
        # !!! Make sure the esm circuits in this file are the same as the ones in check_ft1
        # and also the flag qubits, the syndrome qubits !!!

        # info for the Steane code [[7,1,3]]
        self.p1 = p1
        self.p2 = p2
        self.pm = pm
        self.pI = self.p1*ridle
        # self.ancillas = [8, 9, 10, 11, 12, 13, 14, 80, 90, 100, 110, 120, 130, 140]
        if cir_id == 'c1_l1' or cir_id == 'c1_l2':
            if cir_id == 'c1_l1':
                self.esm_circuits = esm2(idling)
            elif cir_id == 'c1_l2':
                self.esm_circuits = esm3(idling)
            self.q_synd = [9, 11, 13, 80, 100, 120]
            # q_syndx should be smaller than q_syndz
            self.q_syndx = [9, 11, 13]
            self.q_syndz = [80, 100, 120]
            self.q_flag = [8, 10, 12, 90, 110, 130]
            self.stabilisers = {'X': {9: sp.Pauli(x_set=[1,2,4,5]), 11: sp.Pauli(x_set=[1,3,4,7]), 13: sp.Pauli(x_set=[4,5,6,7])},
                                'Z': {80: sp.Pauli(z_set=[1,2,4,5]), 100: sp.Pauli(z_set=[1,3,4,7]), 120: sp.Pauli(z_set=[4,5,6,7])}}
        elif cir_id == 'c2_l1' or cir_id == 'c2_l2' or cir_id == 'c3_l2':

            if cir_id == 'c2_l1':
                self.esm_circuits = esm4(idling)
                self.q_flag = [9, 13, 14, 90, 130, 140] 

            elif cir_id == 'c2_l2':

                self.esm_circuits = esm5(idling)
                
                self.q_flag = [9, 13, 90, 130] 

            elif cir_id == 'c3_l2':
                self.esm_circuits = esm7(idling)
                
                self.q_flag = [9, 90]


            self.q_synd = [8, 10, 12, 80, 100, 120]
            self.q_syndx = [8, 10, 12]
            self.q_syndz = [80, 100, 120]
            self.stabilisers = {'X': {8: sp.Pauli(x_set=[1,2,4,5]), 10: sp.Pauli(x_set=[1,3,4,7]), 12: sp.Pauli(x_set=[4,5,6,7])},
                                'Z': {80: sp.Pauli(z_set=[1,2,4,5]), 100: sp.Pauli(z_set=[1,3,4,7]), 120: sp.Pauli(z_set=[4,5,6,7])}}
        
        self.init_logs = [sp.Pauli(x_set=[1,2,3,4,5,6,7]), sp.Pauli(z_set=[1,2,3,4,5,6,7])]
        
        
        # Find the decoder
        self.lut_synd, self.lut_flag = Check_FT(cir_id).lut_gen()

        self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}

        # used for nndecoder
        self.ancillas = sorted(self.q_synd + self.q_flag)
        self.num_anc = len(self.ancillas)
        self.datas = [1,2,3,4,5,6,7]
        self.num_data = len(self.datas)

        # define parity check matrix and logical matrix
        self.Hx = np.zeros((len(self.q_syndx), 2 * self.num_data), dtype=np.dtype('b'))
        self.Hz = np.zeros((len(self.q_syndz), 2 * self.num_data), dtype=np.dtype('b'))
        self.E = np.zeros((2, 2 * self.num_data), dtype=np.dtype('b'))
        s = 0
        for key in self.stabilisers['X']:
            for j in self.stabilisers['X'][key].x_set:
                self.Hx[s, j - 1 + self.num_data] = 1
            s += 1

        s = 0
        for key in self.stabilisers['Z']:
            for j in self.stabilisers['Z'][key].z_set:
                self.Hz[s, j - 1] = 1
            s += 1

        for i in self.init_logs[0].x_set:
            self.E[0, i - 1 + self.num_data] = 1
        for i in self.init_logs[1].z_set:
            self.E[1, i - 1] = 1

        # syndromes in the first and second round
        self.synds1 = dict()
        self.synds2 = dict()

        # x and z errors on the data qubits
        self.fnl_errsx = dict()
        self.fnl_errsz = dict()

    def err_synd_lld(self):

        for key in self.datas:
            self.fnl_errsx[key] = 0
            self.fnl_errsz[key] = 0
        
        synd_list = ['0'] * self.num_anc*2

        err,synd1 = self.run_first_round()
        if len(synd1):
            err, synd2   = self.run_second_round_without_corr(err,synd1)
        else:
            synd2 = dict()

        err_fnl = err


        for key in err_fnl.x_set:
            self.fnl_errsx[key] = 1
        for key in err_fnl.z_set:
            self.fnl_errsz[key] = 1

        synd_str = self.set_to_list(synd1,synd2,synd_list)
        fnl_err = np.concatenate((np.fromiter(self.fnl_errsx.values(), dtype=float),np.fromiter(self.fnl_errsz.values(), dtype=float)))
#        fnl_synd = list(self.synds1.values()) + list(self.synds2.values())

#       fnl_err = list(self.fnl_errsx.values()) + list(self.fnl_errsz.values())
        
        return([synd_str,fnl_err])
    
    def err_synd_parallel(self):
        # reset all the syndromes and errs to be 0                                                 

        for key in self.ancillas:
            self.synds1[key] = 0
            self.synds2[key] = 0
        for key in self.datas:
            self.fnl_errsx[key] = 0
            self.fnl_errsz[key] = 0

        corr = sp.Pauli()
        synd2 = set()

        err,synd1 = self.run_first_round()

        if len(synd1):
                err, synd2,corr   = self.run_second_round(err,synd1)
                
        err_fnl = err
        err *= corr
        # run another perfect round to clean the left errors
        synd_fnl = set()
        for i in range(len(self.esm_circuits)):
            subcir = self.esm_circuits[i]
            synd_err = circ2err(subcir, fowler_model(subcir, 0, 0, 0), err, self.ancillas)
            synd_fnl |= synd_err[0]
            err = synd_err[1]
        err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndx)))]
        err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndz)))]

        err_tp = singlelogical_error_index(err, self.init_logs)
    
        for key in synd1:
            self.synds1[key] = 1
        for key in synd2:
            self.synds2[key] = 1

        for key in err_fnl.x_set:
            self.fnl_errsx[key] = 1
        for key in err_fnl.z_set:
            self.fnl_errsz[key] = 1

        fnl_synd = list(self.synds1.values()) + list(self.synds2.values())
        fnl_err = list(self.fnl_errsx.values()) + list(self.fnl_errsz.values())

        return np.array(fnl_synd), np.array(fnl_err), err_tp


    
    def run_lut(self, trials = 1):

        for trial in range(trials):
            corr = sp.Pauli()
            synd2 = set()
            err,synd1 = self.run_first_round()
            if len(synd1): 
                err, synd2,corr   = self.run_second_round(err,synd1)

            err_fnl = err
            err *= corr

            # run another perfect round to clean the left errors
            synd_fnl = set()
            for i in range(len(self.esm_circuits)):
                subcir = self.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, 0, 0, 0), err, self.ancillas)
                synd_fnl |= synd_err[0]
                err = synd_err[1]

            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndx)))]
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndz)))]

            # # check logical errors
            err_tp = singlelogical_error(err, self.init_logs)
            self.errors[err_tp] += 1
            
        return self.errors

    def run_lld(self, trials,ds):
        total_errors = 0

        for trial in range(trials):
            err_synd = self.err_synd_lld()
            total_errors =  self.check_for_logical_error(err_synd[1],err_synd[0] ,ds,total_errors)
        return(total_errors)

    def run_hld(self, trials, ds):
        total_errors = 0 
        for trial in range(trials):

            for key in self.datas:
                self.fnl_errsx[key] = 0
                self.fnl_errsz[key] = 0
            
            synd_list = ['0'] * self.num_anc*2
            corr = sp.Pauli()

            err,synd1 = self.run_first_round()
            if len(synd1):
                err, synd2,corr   = self.run_second_round(err,synd1)
            else:
                synd2 = dict()

            err_fnl = err
            err *= corr

            # run another perfect round to clean the left errors                                   
            synd_fnl = set()
            for i in range(len(self.esm_circuits)):
                subcir = self.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, 0, 0, 0), err, self.ancillas)
                synd_fnl |= synd_err[0]
                err = synd_err[1]
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndx)))]
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndz)))]

            # # check logical errors                                                    
            err_tp = singlelogical_error_index(err, self.init_logs)
            
            for key in err_fnl.x_set:
                self.fnl_errsx[key] = 1
            for key in err_fnl.z_set:
                self.fnl_errsz[key] = 1
            
            if err_fnl != sp.str_pauli('I'):
                synd_str = self.set_to_list(synd1,synd2,synd_list)
                if synd_str in ds:
                    if err_tp != np.argmax(ds[synd_str]):
                        total_errors +=1
                else:
                    total_errors +=1
                
            elif (len(synd1)+len(synd2)) != 0:
                synd_str = self.set_to_list(synd1,synd2,synd_list)
                if synd_str in ds:
                    if err_tp != np.argmax(ds[synd_str]):
                        total_errors +=1
                    elif err_fnl != sp.str_pauli('I'):
                        total_errors +=1

        return(total_errors)


    def run_first_round(self):
        err = sp.Pauli()
        synd1 = set()
        for i in range(len(self.esm_circuits)):
                subcir = self.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, 
                                                         self.pI), err, self.ancillas)
                synd1 |= synd_err[0]
                err = synd_err[1]
                if len(synd1):
                    # if there is a syndrome or a flag, then stop this round              
                    break
        return(err,synd1)
        
    def run_second_round_without_corr(self,err,synd1):
        synd2 = set()
        for i in range(len(self.esm_circuits)):
            subcir = self.esm_circuits[i]
            synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p2,
                                                     self.pm, self.pI), err, self.ancillas)
            synd2 |= synd_err[0]
            err = synd_err[1]
        return(err, synd2)
 
    def run_second_round(self,err,synd1):
        corr = sp.Pauli()
        synd2 = set()
        for i in range(len(self.esm_circuits)):
            subcir = self.esm_circuits[i]
            synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, 
                                                     self.pm, self.pI), err, self.ancillas)
            synd2 |= synd_err[0]
            err = synd_err[1]
                
        # Choose lut decoder based on whether there is a flag, then find corrections
        flag_qs = synd1 & set(self.q_flag)
        synd_qx = sorted(synd2 & set(self.q_syndx))
        synd_qz = sorted(synd2 & set(self.q_syndz))
        if len(flag_qs):
            corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qx)]
            corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qz)]
        else:
            corr *= self.lut_synd[tuple(synd_qz)]
            corr *= self.lut_synd[tuple(synd_qx)]
        
        return(err, synd2,corr)
        
    def set_to_list(self,synd1,synd2,synd_list):
        for i in range(self.num_anc):
            if self.ancillas[i] in synd1:
                synd_list[i] = '1'

        for j in range(self.num_anc):
            if self.ancillas[j] in synd2:
                synd_list[j] = '1'

        synd_str = ''.join(synd_list)

        return(synd_str)

    def check_for_logical_error(self,errs, synd,ds,c):
        errs = errs.ravel()
        
        if synd in ds:
            sample = np.round(ds[synd])
        else:
            sample = np.zeros((self.num_data*2))
    
        left_errs = (sample+errs)%2
        syndx_fnl = self.Hx.dot(left_errs)%2
        syndz_fnl = self.Hz.dot(left_errs)%2
        errors = self.E.dot(left_errs)%2
    
        if np.any(syndx_fnl):
            if errors[0] == 0:
                c += 1
            elif np.any(syndz_fnl):
                if errors[1] == 0: 
                    c += 1
            else:
                if errors[1]:
                    c += 1

        elif np.any(syndz_fnl):
            if errors[1] == 0: 
                c += 1
            elif errors[0]:
                c += 1
        else:
            if np.any(errors):
                c += 1
        return(c)


def circ2err(circ, err_model, err, anc):
    """
    Add errors to the circuit circ
    """
    synd = set()
    for stp, mdl in list(zip(circ, err_model)):
        # run timestep, then sample
        new_synds, err = cm.apply_step(stp, err)
        err *= product(_.sample() for _ in mdl)
        synd |= new_synds[1]
    
    # last round of circuit, because there are n-1 errs, n gates
    new_synds, err = cm.apply_step(circ[-1], err)
    synd |= new_synds[1]
    # remove remaining errors on ancilla qubits before append
    err.prep(anc)

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
    returns a single letter recording the resulting logical error (may be I,
    X, Y or Z)
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

    return anticom_dict[ ( x_com, z_com ) ]

def singlelogical_error_index(err_fnl, corr_logs):
    """
    returns a single letter recording the resulting logical error (may be I,
    X, Y or Z)
    """
    x_log = corr_logs[0]
    z_log = corr_logs[1]
    anticom_dict = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    x_com, z_com = x_log.com(err_fnl), z_log.com(err_fnl)

    return anticom_dict[ ( x_com, z_com ) ]


def check_lut(p1, p2, pm, cir_id, trials, idling=False, ridle=0):
    x = Steane_FT(p1, p2, pm, cir_id, idling=idling, ridle=ridle)
    err = x.run_lut(trials)
    return err

def check_hld(p1, p2, pm, cir_id, trials, idling, ridle, ds):
    x = Steane_FT(p1, p2, pm, cir_id, idling=idling, ridle=ridle)
    err = x.run_hld(trials,ds)
    return err

if __name__ == '__main__':
    err = check_lut(0.001, 0.001, 0.001, 'c1_l2', 1000, idling=False, ridle=0)
    print(err)
