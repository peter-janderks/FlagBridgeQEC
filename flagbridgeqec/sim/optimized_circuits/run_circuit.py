import sparse_pauli as sp
from flagbridgeqec.circuits import cir_steane_4a, cir_steane_5a, cir_steane_6a, cir_s17_13
from flagbridgeqec.sim.optimized_circuits.check_ft1_parallel import Check_FT
#import check_ft1_parallel as cf
from flagbridgeqec.utils import error_model_2 as em
from flagbridgeqec.circuits.read_test_circuits import compile_circuit
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

        self.esm_circuits = self.esm_circuits_steane(cir_id,self.pI)

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
            
        elif cir_id == 'IBM_11':

            self.q_synd = [8,9,10,80,90,100]
            self.q_flag = [11, 110]
            self.q_syndx = [8, 9, 10]
            self.q_syndz = [80, 90, 100]
            self.x_ancillas = set((80,90,100,110))
            self.z_ancillas = set((8,9,10,11))

        elif cir_id == 'IBM_12':
            self.q_synd = [8,10,12,80,100,120]
            self.q_flag = [9,11,90,110]
            self.q_syndx = [80,100,120]
            self.q_syndz = [8,10,12]

        elif cir_id == 'IBM_13':
            self.q_flag = [8,10,12,80,100,120]
            self.q_synd = [130, 110, 90, 9, 11, 13]
            self.q_syndx = [130,110,90]
            self.q_syndz = [9, 11, 13]

        elif cir_id == 's17_33':
            self.q_synd = [8,10,11, 80,100,110]
            self.q_flag = [90, 120, 130, 9, 12, 13]
            self.q_syndx = [80, 100, 110]
            self.q_syndz = [8, 10, 11]

        elif cir_id == 's17_222':
            self.q_synd = [9,11,13,90,110,130]
            self.q_flag = [80, 100, 120, 8, 10, 12]
            self.q_syndx = [9, 11, 13]
            self.q_syndz = [90, 110, 130]
            self.x_ancillas = set((80,90,100,110,120))
            self.z_ancillas = set((8,9,10,11,12))

        elif cir_id == 'a4_L2_split':
            Print('test')

        self.cir_index = cir_id

        self.ancillas = self.q_synd + self.q_flag
        self.num_anc = len(self.ancillas)
        self.lut_synd, self.lut_flag = Check_FT(cir_id).lut_gen()

        self.datas = [1,2,3,4,5,6,7]

        self.stabilisers = {'X': {9: sp.Pauli(x_set=[1,2,4,5]), 
                                  11: sp.Pauli(x_set=[1,3,4,7]),
                                  13: sp.Pauli(x_set=[4,5,6,7])},
                            'Z': {80: sp.Pauli(z_set=[1,2,4,5]), 
                                  100: sp.Pauli(z_set=[1,3,4,7]), 
                                  120: sp.Pauli(z_set=[4,5,6,7])}}
        self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}
        self.init_logs = [sp.Pauli(x_set=[1,2,3,4,5,6,7]), sp.Pauli(z_set=[1,2,3,4,5,6,7])]
        # syndromes in the first and second round                                                    
        self.synds1 = dict()
        self.synds2 = dict()

        # x and z errors on the data qubits                                                          
        self.fnl_errsx = dict()
        self.fnl_errsz = dict()


    def esm_circuits_steane(self,cir_id, idling=False):
        if cir_id == '6a':
            esm_circuits = []
            esm_circuits.extend([cir_steane_6a(1,idling),cir_steane_6a(2,idling),cir_steane_6a(3,idling),cir_steane_6a(4,idling)])
        elif cir_id == '4a':
            esm_circuits = []
            esm_circuits.extend([cir_steane_4a(1,idling),cir_steane_4a(2,idling),cir_steane_4a(3,idling), cir_steane_4a(4,idling), cir_steane_4a(5,idling)])
        elif cir_id == '5a':
            esm_circuits = []
            esm_circuits.extend([cir_steane_5a(1,idling),cir_steane_5a(2,idling),cir_steane_5a(3,idling),cir_steane_5a(4,idling)])
        elif cir_id == 's17_13':
            esm_circuits = []
            esm_circuits.extend([cir_s17_13(1,idling),cir_s17_13(2,idling),cir_s17_13(3,idling),cir_s17_13(4,idling)])

        elif cir_id == 'IBM_11':
            esm_circuits = compile_circuit('IBM_11',idling)
            self.x_ancillas = set((80,90,100,110))
            self.z_ancillas = set((8,9,10,11))

        elif cir_id == 'IBM_12':
            esm_circuits = compile_circuit('IBM_12',idling)
            self.x_ancillas = set((80,90,100,110,120))
            self.z_ancillas = set((8,9,10,11,12))

        elif cir_id == 'IBM_13':
            esm_circuits = compile_circuit('IBM_13',idling)
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))
            
        elif cir_id == 's17_33':
            esm_circuits = compile_circuit('s17_33',idling)
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))

        elif cir_id == 's17_222':
            esm_circuits = compile_circuit('s17_222',idling)
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))

        elif cir_id == 'a4_L2_split':
            esm_circuits = []
            esm_circuits.extend([a4_L2_split(1),cir_steane_5a(2),cir_steane_5a(3),cir_steane_5a(4,)])

        return(esm_circuits)
        


    def run(self, trials=1):
        total_errors = 0 
        no_synd = 0

        for trial in range(trials):
            corr = sp.Pauli()
            synd2 = set()
            err,synd1, break_point = self.run_first_round()
            
            if len(synd1):
                err, synd2,corr = self.run_second_round(err,synd1,break_point)
            else: 
                no_synd +=1

            err_fnl = err
            err *= corr

            # run another perfect round to clean the left errors                     
            synd_fnl = set()
            for i in range(2):
                subcir = self.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, 0, 0, 0), err, self.ancillas)
                synd_fnl |= synd_err[0]
                err = synd_err[1]
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndx)))]
            err *= self.lut_synd[tuple(sorted(synd_fnl & set(self.q_syndz)))]

            # # check logical errors 
            err_tp = self.singlelogical_error(err, self.init_logs)
            self.errors[err_tp] += 1
            

        return (self.errors)

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

        err,synd1,breakpoint = self.run_first_round()

        if len(synd1):
                err, synd2,corr  = self.run_second_round(err,synd1,breakpoint)

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

        err_tp = self.singlelogical_error_index(err, self.init_logs)

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
    def run_first_round(self):
        err = sp.Pauli()
        synd1 = set()
        for i in range(2):

            subcir = self.esm_circuits[i]
            synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p1, self.p1, self.pI), err, self.ancillas)
            synd1 |= synd_err[0]
            err = synd_err[1]
            if len(synd1):
                # if there is a syndrome or a flag, then stop this round  
                break
        return(err,synd1,i)

    def run_second_round(self,err,synd1,i):
        corr = sp.Pauli()
        synd2 = set()
        subcir = self.esm_circuits[i+2]

        synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p1,
                                                 self.p1, self.pI), err, self.ancillas)
        synd2 |= synd_err[0]
        err = synd_err[1]

            # Choose lut decoder based on whether there is a flag, then find corrections                                
        flag_qs = synd1 & set(self.q_flag)
        synd_qx = sorted(synd2 & set(self.q_syndx))
        synd_qz = sorted(synd2 & set(self.q_syndz))

        if len(flag_qs):
            try:
                
                corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qx)]
                corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qz)]
            except KeyError:
                pass
        else:
            try:
                corr *= self.lut_synd[tuple(synd_qz)]
                corr *= self.lut_synd[tuple(synd_qx)]
                
            except KeyError:
                pass

        return(err, synd2, corr)

    def run_hld(self, trials, ds):
        no_error = 0
        total_errors = 0
        for trial in range(trials):

            for key in self.datas:
                self.fnl_errsx[key] = 0
                self.fnl_errsz[key] = 0

            synd_list = ['0'] * self.num_anc*2
            synd_zeros = '0' * self.num_anc*2
            corr = sp.Pauli()

            err,synd1,breakpoint = self.run_first_round()
            if len(synd1):
                err, synd2,corr = self.run_second_round(err,synd1,breakpoint)
            else:
                synd2 = dict()
                no_error += 1

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
            err_tp = self.singlelogical_error_index(err, self.init_logs)

            for key in err_fnl.x_set:
                self.fnl_errsx[key] = 1
            for key in err_fnl.z_set:
                self.fnl_errsz[key] = 1

            synd_str = self.set_to_list_with_syndromes(synd1,synd2,synd_list)
            if synd_str != synd_zeros:
                if synd_str in ds:
                    if err_tp != np.argmax(ds[synd_str]):
                        total_errors +=1
                elif err_tp != 0:
                    total_errors +=1
                    print(synd1,'synd1')
                    print(synd2,'synd2')
                    print(synd_str,'synd_str')
                    print('not in dataset')

        return(total_errors,no_error)

    def singlelogical_error_index(self, err_fnl, corr_logs):
        """                                                                                       
        returns a single letter recording the resulting logical error (may be I, X, Y or Z)
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

    def set_to_list_with_syndromes(self,synd1,synd2,synd_list):
        for i in range(self.num_anc):
            if self.ancillas[i] in synd1:
                synd_list[i] = '1'

        for j in range(self.num_anc):
            if self.ancillas[j] in synd2:
                synd_list[j+self.num_anc] = '1'

        synd_str = ''.join(synd_list)
        return(synd_str)


    def run_old(self, trials = 1):

        total_errors = 0 
        no_synd = 0
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
                        pass

                else:
                    try:

                        corr *= self.lut_synd[tuple(synd_qz)]
                        corr *= self.lut_synd[tuple(synd_qx)]
                    except KeyError:
                        pass
            else:
                no_synd +=1

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
                        pass
                        
                else:
                    try:
                        corr *= self.lut_synd[tuple(synd_qz)]
                        corr *= self.lut_synd[tuple(synd_qx)]
                    except KeyError:
                        pass
                        #                            print('not in lut')

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
                total_errors +=1
        return(total_errors)
        
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

def check_hld(p1, p2, pm, cir_id, trials, idling, ridle):
        x = FT_protocol(p1, cir_id, p_idling=ridle)
        err = x.run_hld(trials,ds)
        return err

def check(per,trials, cir_id,p_idling):
    x = FT_protocol(per,cir_id,p_idling)
    err = x.run(trials)
    return(err)

if __name__ == '__main__':
    x = FT_protocol(0.0005,cir_id='s17_222',p_idling=0)
    error = x.run(1000)
    print(error)
