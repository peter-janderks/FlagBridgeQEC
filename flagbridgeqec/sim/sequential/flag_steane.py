import sparse_pauli as sp
from check_ft1 import Check_FT
import check_ft1 as cf
from flagbridgeqec.utils import error_model_2 as em
import circuit_metric as cm
from operator import mul
from functools import reduce
import numpy as np
from flagbridgeqec.circuits import esmx_anc, esmz_anc, esmxs_anc3, esmzs_anc3, esmxs_anc4, esmzs_anc4

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
        print(cir_id)
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


    def err_synd(self):
        # reset all the syndromes and errs to be 0
        for key in self.ancillas:
            self.synds1[key] = 0
            self.synds2[key] = 0
        for key in self.datas:
            self.fnl_errsx[key] = 0
            self.fnl_errsz[key] = 0

        err = sp.Pauli()
        corr = sp.Pauli()
        synd1 = set()
        synd2 = set()
        for i in range(len(self.esm_circuits)):
            subcir = self.esm_circuits[i]
            s_e = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, pI=0), err, self.ancillas)
            synd1 |= s_e[0]
            err = s_e[1]
            if len(synd1):
                # if there is a syndrome or a flag, then stop this round
                break
        # if there is a syndrome or a flag, after stop this round, start a full new round for all stabs
        
        if len(synd1):
            for i in range(len(self.esm_circuits)):
                subcir = self.esm_circuits[i]
                s_e = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, pI=0), err, self.ancillas)
                synd2 |= s_e[0]
                err = s_e[1]

        # for key in synd1:
        if len(synd1 & set(self.q_flag)):
            for key in synd1 & set(self.q_flag):
                self.synds1[key] = 1
        for key in synd2:
            self.synds2[key] = 1

        for key in err.x_set:
            self.fnl_errsx[key] = 1
        for key in err.z_set:
            self.fnl_errsz[key] = 1

        fnl_synd = list(self.synds1.values()) + list(self.synds2.values())
        fnl_err = list(self.fnl_errsx.values()) + list(self.fnl_errsz.values())

        return np.array(fnl_synd), np.array(fnl_err)


    def err_synd2(self):
        # reset all the syndromes and errs to be 0
        for key in self.ancillas:
            self.synds1[key] = 0
            self.synds2[key] = 0
        for key in self.datas:
            self.fnl_errsx[key] = 0
            self.fnl_errsz[key] = 0

        err = sp.Pauli()
        corr = sp.Pauli()
        synd1 = set()
        synd2 = set()
        for i in range(len(self.esm_circuits)):
            subcir = self.esm_circuits[i]
            s_e = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, pI=0), err, self.ancillas)
            synd1 |= s_e[0]
            err = s_e[1]
            
        for i in range(len(self.esm_circuits)):
            subcir = self.esm_circuits[i]
            s_e = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, pI=0), err, self.ancillas)
            synd2 |= s_e[0]
            err = s_e[1]

        for key in synd1:
            self.synds1[key] = 1
        for key in synd2:
            self.synds2[key] = 1

        for key in err.x_set:
            self.fnl_errsx[key] = 1
        for key in err.z_set:
            self.fnl_errsz[key] = 1

        fnl_synd = list(self.synds1.values()) + list(self.synds2.values())
        fnl_err = list(self.fnl_errsx.values()) + list(self.fnl_errsz.values())

        return np.array(fnl_synd), np.array(fnl_err)


    def run(self, trials = 1):

        for trial in range(trials):
            err = sp.Pauli()
            corr = sp.Pauli()
            synd1 = set()
            synd2 = set()
            for i in range(len(self.esm_circuits)):
                subcir = self.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, self.pI), err, self.ancillas)
                synd1 |= synd_err[0]
                err = synd_err[1]
                if len(synd1):
                    # if there is a syndrome or a flag, then stop this round
                    break
            # if there is a syndrome or a flag, after stop this round, start a full new round for all stabs
            if len(synd1):
                for i in range(len(self.esm_circuits)):
                    subcir = self.esm_circuits[i]
                    synd_err = circ2err(subcir, fowler_model(subcir, self.p1, self.p2, self.pm, self.pI), err, self.ancillas)
                    synd2 |= synd_err[0]
                    err = synd_err[1]
                # Choose lut decoder based on whether there is a flag, then find corrections
                flag_qs = synd1 & set(self.q_flag)
                synd_qx = sorted(synd2 & set(self.q_syndx))
                synd_qz = sorted(synd2 & set(self.q_syndz))
                if len(flag_qs):
                    # print(self.lut_flag)
                    # print(flag_qs)
                    # print(len(self.lut_flag))
                    # print( self.lut_flag[tuple(flag_qs)].keys())
                    # print(tuple(synd_qx))
                    # print(tuple(synd_qz))
                    corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qx)]
                    corr *= self.lut_flag[tuple(flag_qs)][tuple(synd_qz)]
                else:
                    # print(self.lut_synd.keys())
                    # print(tuple(synd_qx))
                    # print(tuple(synd_qz))
                    corr *= self.lut_synd[tuple(synd_qz)]
                    corr *= self.lut_synd[tuple(synd_qx)]

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

            # print(self.lut_synd)
            # print(synd_fnl)
            # # check logical errors
            # print(err)
            err_tp = singlelogical_error(err, self.init_logs)
            # if err_tp != 'I':
            #     # print(self.lut_flag, self.lut_synd)
            #     print('synds are')
            #     print(synd1, synd2)
            #     print('fnl errors are', err_fnl)
            #     print('err is', err)
            #     print('corr is', corr)
            #     # break
            self.errors[err_tp] += 1

        return self.errors

                            

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


def esm2(idling=False):
    """
    syndrome extractor circuit for steane circuit 2, two ancillas per face, 6 checks in series
    """
    cirs = []
    cirs.append(cf.esmx_anc([4,5,6,7], [12, 13], idling=idling))
    cirs.append(cf.esmx_anc([1,2,4,5], [8, 9], idling=idling))
    cirs.append(cf.esmx_anc([1,3,4,7], [10, 11], idling=idling))
    cirs.append(cf.esmz_anc([4,5,6,7], [120, 130], idling=idling))
    # cirs.append(esmz_anc([4,5,6,7], [12, 13, 14], idling=idling)
    cirs.append(cf.esmz_anc([1,2,4,5], [80, 90], idling=idling))
    cirs.append(cf.esmz_anc([1,3,4,7], [100, 110], idling=idling))
    return cirs
    

def esm3(idling=False):
    """
    syndrome extractor circuit for steane circuit 2, two ancillas per face, 6 checks in series
    """
    cirs = []
    cirs.append(cf.esmx_anc([4,5,6,7], [12, 13], idling=idling, db=3))
    cirs.append(cf.esmx_anc([1,2,4,5], [8, 9], idling=idling))
    cirs.append(cf.esmx_anc([1,3,4,7], [10, 11], idling=idling))
    cirs.append(cf.esmz_anc([4,5,6,7], [120, 130], idling=idling, db=3))
    cirs.append(cf.esmz_anc([1,2,4,5], [80, 90], idling=idling))
    cirs.append(cf.esmz_anc([1,3,4,7], [100, 110], idling=idling))
    return cirs


def esm4(idling=False):
    """
    syndrome extractor circuit for steane circuit 5, two checks in parallel
    """
    cirs = []
    cirs.append(cf.esmxs_anc3( [3,7,2,5], [3,7,1,4], [8, 10, 9], idling=idling))
    cirs.append(cf.esmx_anc([4,5,6,7], [12, 13, 14], idling=idling))
    cirs.append(cf.esmzs_anc3( [3,7,2,5], [3,7,1,4], [80, 100, 90], idling=idling))
    cirs.append(cf.esmz_anc([4,5,6,7], [120, 130, 140], idling=idling))

    return cirs


def esm5(idling=False):
    """
    syndrome extractor circuit for steane circuit 5, two checks in parallel
    """
    cirs = []
    cirs.append(cf.esmxs_anc3( [3,7,2,5], [3,7,1,4], [8, 10, 9], idling=idling))
    cirs.append(cf.esmx_anc([4,5,6,7], [13, 12], idling=idling))
    cirs.append(cf.esmzs_anc3( [3,7,2,5], [3,7,1,4], [80, 100, 90], idling=idling))
    cirs.append(cf.esmz_anc([4,5,6,7], [120, 130], idling=idling))

    return cirs

def esm7(idling=False):
    """
    syndrome extractor circuit for steane circuit 5, two checks in parallel
    """
    cirs = []
    cirs.append(cf.esmxs_anc4([4,1,5,2], [4,1,3,7], [4,5,6,7], [8, 10, 12, 9], chao=False, idling=idling))
    cirs.append(cf.esmzs_anc4([4,1,5,2], [4,1,3,7], [4,5,6,7], [80, 100, 120, 90], chao=False, idling=idling))

    return cirs

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


def check(p1, p2, pm, cir_id, trials, idling=False, ridle=0):
    x = Steane_FT(p1, p2, pm, cir_id, idling=idling, ridle=ridle)
    err = x.run(trials)
    return err

if __name__ == '__main__':
    err = check(0.001, 0.001, 0.001, 7, 10, idling=False, ridle=0)
    print(err)
