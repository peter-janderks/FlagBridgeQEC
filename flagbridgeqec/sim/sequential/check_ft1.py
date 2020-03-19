import circuit_metric as cm
import sparse_pauli as sp

from flagbridgeqec.circuits import esmx_anc, esmz_anc, esmxs_anc3, esmzs_anc3, esmxs_anc4, esmzs_anc4

flatten = lambda l: [item for sublist in l for item in sublist]

class Check_FT(object):
    def __init__(self, cir_index=2, steane=True):

        self.cir_index = cir_index
        
        if cir_index == 'c1_l1' or cir_index == 'c1_l2':
            # for cir_steane2,3
            self.q_synd = [9, 11, 13, 80, 100, 120]
            self.q_flag = [8, 10, 12, 90, 110, 130]
        elif cir_index == 'c2_l1' or cir_index == 'c2_l2':
            # for cir_steane4,5,6
            self.q_synd = [8, 10, 12, 80, 100, 120]
            if cir_index == 'c2_l1':
                self.q_flag = [9, 13, 14, 90, 130, 140] 
            else:
                self.q_flag = [9, 13, 90, 130] 
        elif cir_index == 'c3_l2':
            # for cir_steane7,8
            self.q_synd = [8, 10, 12, 80, 100, 120]
            self.q_flag = [9, 90]

        self.ancillas = self.q_synd + self.q_flag

        self.init_stabs = [sp.Pauli(x_set=[1,2,4,5]), sp.Pauli(x_set=[4,5,6,7]), sp.Pauli(x_set=[1,3,4,7]), 
                           sp.Pauli(z_set=[1,2,4,5]), sp.Pauli(z_set=[4,5,6,7]), sp.Pauli(z_set=[1,3,4,7])]
        self.init_logs = [sp.Pauli(x_set=[1,2,3,4,5,6,7]), sp.Pauli(z_set=[1,2,3,4,5,6,7])]
        self.stabilisers = {'X': {8: sp.Pauli(x_set=[1,2,4,5]), 10: sp.Pauli(x_set=[1,3,4,7]), 12:  sp.Pauli(x_set=[4,5,6,7])},
                            'Z': {80: sp.Pauli(z_set=[1,2,4,5]), 100: sp.Pauli(z_set=[1,3,4,7]), 120:  sp.Pauli(z_set=[4,5,6,7])}}
        
        self.two_error = ['IX', 'IZ', 'IY', 'XI', 'ZI', 'YI', 'XX', 'XZ', 'XY', 'ZX', 'ZZ', 'ZY', 'YX', 'YZ', 'YY']
        self.one_error = ['IX', 'IZ', 'IY']
        
    def ft_circuit(self, tp='XZ'):
        if self.cir_index == 'c1_l2':
            timestps = steane_c1_l2(tp)
        elif self.cir_index == 'c1_l1':
            timestps = steane_c1_l1(tp)
        elif self.cir_index == 'c2_l1':
            timestps = steane_c2_l1(tp)
        elif self.cir_index == 'c2_l2':
            timestps = steane_c2_l2(tp)
        elif self.cir_index == 'c3_l2':
            timestps = steane_c3_l2(tp)
        return timestps

    def lut_gen(self):
        # generate the look-up-table decocder for steane code
        
        extractor_x = flatten(self.ft_circuit(tp='X'))
        extractor_z = flatten(self.ft_circuit(tp='Z'))
        print('timesteps number', len(self.ft_circuit(tp='X'))+len(self.ft_circuit(tp='Z')))
        print('gate number', len(extractor_x)+len(extractor_z))
        cnot_n = 0
        cnot_nd = 0
        for i in extractor_z + extractor_x:
            if i[0] == 'CNOT':
                cnot_n += 1
                if len(set(self.ancillas) & set(i[1:])) != 2:
                    cnot_nd += 1
        print('cnot number', cnot_n)
        print('cnot on data number', cnot_nd)

        err_synds = circuit_2err2(self.ancillas, extractor_z, extractor_x, two_err=self.two_error, one_err=None)
        lut_synd = dict()
        lut_flag = dict()
        for synd in err_synds.keys():
            if len(synd[0]) + len(synd[1]) == 0:
                pass
            else:
                synd_r1 = set(synd[0]) | set(synd[1])
                synd_r2 = set(synd[2]) | set(synd[3])
                err_min = self.init_logs[0]
                for qbt in err_synds[synd]:
                    err = sp.Pauli(x_set=qbt[0], z_set=qbt[1])
                    if err.weight() < err_min.weight():
                        err_min = err
                err_minx = self.init_logs[0]
                err_minz = self.init_logs[1]
                for qbt in err_synds[synd]:
                    err = sp.Pauli(x_set=qbt[0])
                    if err.weight() < err_minx.weight():
                        err_minx = err
                for qbt in err_synds[synd]:
                    err = sp.Pauli(z_set=qbt[1])
                    if err.weight() < err_minz.weight():
                        err_minz = err
                synd_r2 = [set(synd[2]),set(synd[3])]
                err_mins = [err_minz, err_minx]
                if synd_r1 & set(self.q_flag):
                    # if flag1 flags
                    flag1 = tuple(synd_r1 & set(self.q_flag))
                    if flag1 not in lut_flag.keys():
                        lut_flag[flag1] = dict()

                    for i in range(len(synd_r2)):
                        flag_key = tuple(sorted(synd_r2[i]))
                        if flag_key not in lut_flag[flag1]:
                            lut_flag[flag1][flag_key] = err_mins[i]
                        else:
                            if err_mins[i] != lut_flag[flag1][flag_key]:
                                for log in self.init_logs:
                                    if (err_mins[i] * lut_flag[flag1][flag_key]).com(log):
                                        print('synd is', str(flag_key))
                                        print(err_mins[i])
                                        print(lut_flag[flag1][flag_key])
                                        print(err_synds)
                                        raise('Possibly not distingushed syndromes')
                            else:
                                pass
                elif synd_r1 & set(self.q_synd):
                    
                    for i in range(len(synd_r2)):
                        synd_key = tuple(sorted(synd_r2[i]))
                        if synd_key not in lut_synd:
                            lut_synd[synd_key] = err_mins[i]
                        else:
                            for log in self.init_logs:
                                if err_mins[i] != lut_synd[synd_key]:
                                    if (err_mins[i] * lut_synd[synd_key]).com(log):
                                        print(err_synds)
                                        print('synd is', synd_key)
                                        print(err_mins[i])
                                        print(lut_synd[synd_key])
                                        raise('Possibly not distingushed syndromes')

        # Add other pairs of syndrome:error for flag cases
        for flag in lut_flag.keys():
            for synd in lut_synd.keys():
                if synd not in lut_flag[flag].keys():
                    lut_flag[flag][synd] = lut_synd[synd]

        return lut_synd, lut_flag

    def run(self):

        extractor_x = flatten(self.ft_circuit(tp='X'))
        extractor_z = flatten(self.ft_circuit(tp='Z'))
        err_synds = circuit_2err2(self.ancillas, extractor_z, extractor_x, two_err=self.two_error, one_err=None)
        ft_value = 0
        for synd_list in err_synds:
            for i in err_synds[synd_list]:
                if len(i[0]) == 2 or len(i[1]) == 2:
                    
                    if (set(synd_list[0]) | set(synd_list[1])) & set(self.q_flag):
                        pass
                    else:
                        print('synds are', synd_list)
                        print('errs are', i)
                        raise('two errors but no flags')
                        # print((set(synd_list[0]) | set(synd_list[1])) & set(self.q_flag))
               
            if len(err_synds[synd_list]) > 1:
                for i in range(len(err_synds[synd_list])-1):
                    for j in range(i+1, len(err_synds[synd_list])):
                        err1 = sp.Pauli()
                        err2 = sp.Pauli()
                        err1 *= sp.Pauli(err_synds[synd_list][i][0], err_synds[synd_list][i][1])
                        err2 *= sp.Pauli(err_synds[synd_list][j][0], err_synds[synd_list][j][1])

                        # check the fault-tolerance by checking whether the multiplication of two errors 
                        # is a logical. This should be ok for esm circuit checking, but how about logical gates
                        for stab in self.init_stabs:
                            if (err1 * err2).com(stab):
                                ft_value = 0
                                print('One case that errors are going to be a logical')
                                print(synd_list, err_synds[synd_list])
                                return ft_value
                            else:
                                pass

                        if (err1 * err2).com(self.init_logs[0]):
                            ft_value = 0
                            print('One case that could cause a logical error')
                            print(synd_list, err_synds[synd_list])
                            return ft_value
                        elif (err1 * err2).com(self.init_logs[1]):
                            ft_value = 0
                            print('One case that could cause a logical error')
                            print(synd_list, err_synds[synd_list])
                            return ft_value
                        else:
                            ft_value = 1

            else:
                ft_value = 1

        return ft_value

def add_idling(timesteps, anc_set, data_set):
    dat_locs = set(anc_set + data_set)
    for step in timesteps:
        step.extend([('I', q) for q in dat_locs - support(step)])
    return timesteps


def steane_c1_l2(tp='XZ'):
    """
    The circuit to perform each check for the Steane code using two ancillas per face
    """
    timesteps = []
    if 'Z' in tp:
        timesteps += esmz_anc([4,5,6,7], [120, 130])
        timesteps += esmz_anc([1,2,4,5], [80, 90])
        timesteps += esmz_anc([1,3,4,7], [100, 110])
    if 'X' in tp:
        timesteps += esmx_anc([4,5,6,7], [12, 13])
        timesteps += esmx_anc([1,2,4,5], [8, 9])
        timesteps += esmx_anc([1,3,4,7], [10, 11])

    return timesteps


def steane_c1_l1(tp='XZ'):
    """
    The circuit to perform each check for the Steane code using two ancillas per face
    """
    timesteps = []
    if 'Z' in tp:
        timesteps += esmz_anc([4,5,6,7], [120, 130], db=3)
        timesteps += esmz_anc([1,2,4,5], [80, 90])
        timesteps += esmz_anc([1,3,4,7], [100, 110])
    if 'X' in tp:
        timesteps += esmx_anc([4,5,6,7], [12, 13], db=3)
        timesteps += esmx_anc([1,2,4,5], [8, 9])
        timesteps += esmx_anc([1,3,4,7], [10, 11])

    return timesteps


def steane_c2_l1(tp='XZ'):
    """
    The circuit to perform two X or two Z checks in parallel for the Steane code using 3+2 ancillas 
    """
    timesteps = []
    if 'Z' in tp:
        timesteps += esmzs_anc3( [3,7,2,5], [3,7,1,4], [80, 100, 90])
        timesteps += esmz_anc([4,5,6,7], [120, 130, 140])
    
    if 'X' in tp:
        timesteps += esmxs_anc3( [3,7,2,5], [3,7,1,4], [8, 10, 9])
        timesteps += esmx_anc([4,5,6,7], [12, 13, 14])

    return timesteps


def steane_c2_l2(tp='XZ'):
    """
    The circuit to perform two X or two Z checks in parallel for the Steane code using 3+2 ancillas 
    """
    timesteps = []
    if 'Z' in tp:
        timesteps += esmzs_anc3( [3,7,2,5], [3,7,1,4], [80, 100, 90])
        timesteps += esmz_anc([4,5,6,7], [120, 130])
    
    if 'X' in tp:
        timesteps += esmxs_anc3( [3,7,2,5], [3,7,1,4], [8, 10, 9])
        timesteps += esmx_anc([4,5,6,7], [13, 12])

    return timesteps

def steane_c3_l2(tp='XZ'):
    """
    The circuit to perform three X or two Z checks in parallel for the Steane code using 4 ancillas 
    """
    timesteps = []
    if 'Z' in tp:
        timesteps += esmzs_anc4([4,1,5,2], [4,1,3,7], [4,5,6,7], [80, 100, 120, 90])
    
    if 'X' in tp:
        timesteps += esmxs_anc4([4,1,5,2], [4,1,3,7], [4,5,6,7], [8, 10, 12, 9])

    return timesteps

def esm_apply_err(extractor, anc, errors, rg=0, bk=False):
        synd = set()
        for j in range(rg, len(extractor)):
            if extractor[j][0] == 'P':
         #       print(extractor[j])
                # remove the errors on the ancillas
                errors.prep(anc)
                if bk:
                    break
            else:
                new_synds, errors = cm.apply_step([extractor[j]], errors)
                synd |= new_synds[1]    

        errors.prep(anc)  
        return synd, errors

def add2check(extractor, err_synd, two_err, anc, check_tp, alph_order=True):
    tp = check_tp
    ntp = 'XZ'.replace(tp, '')

    for i in range(len(extractor[tp])):
        if len(list(extractor[tp][i])) == 3:
            # if it is a two-qubit gate
            for err in two_err:
                # add 16 kinds of pauli errors one by one
                error = sp.Pauli()
                if err[0] == 'X' or err[0] == 'Y':
                    error *= sp.Pauli({int(list(extractor[tp][i])[1])}, {})
                if err[0] == 'Z' or err[0] == 'Y': 
                    error *= sp.Pauli({}, {int(list(extractor[tp][i])[1])})
                if err[1] == 'X' or err[1] == 'Y':
                    error *= sp.Pauli({int(list(extractor[tp][i])[-1])}, {})
                if err[1] == 'Z' or err[1] == 'Y': 
                    error *= sp.Pauli({}, {int(list(extractor[tp][i])[-1])})
                    
                synd1 = {'X': [], 'Z': []}


                synd1[tp], error = esm_apply_err(extractor[tp], anc, error, rg=i+1, bk=True)
                # The execution breaks here because we assume if we see a syndrome or a flag, 
                # we will stop this round and start a full of new round
                # if extractor[ntp]:
                #     synd1[ntp], error = esm_apply_err(extractor[ntp], anc, error)
                # synd1[ntp] = set()
                
                # apply these errors to the extractors to get the syndromes of next perfect round
                synd2 = {'X': [], 'Z': []}
                if alph_order:
                    synd2[tp], error = esm_apply_err(extractor[tp], anc, error)
                    if extractor[ntp]:
                        synd2[ntp], error = esm_apply_err(extractor[ntp], anc, error)
                else:
                    synd2[ntp], error = esm_apply_err(extractor[ntp], anc, error)
                    if extractor[tp]:
                        synd2[tp], error = esm_apply_err(extractor[tp], anc, error)
                
                # define a tuple here, the first (last) two elements are the synds from the first(second) round of esm
                synd_list = (tuple(synd1['X']), tuple(synd1['Z']), tuple(synd2['X']), tuple(synd2['Z']))
                if synd_list in err_synd.keys():
                    if (error.x_set, error.z_set) in err_synd[synd_list]:
                        pass
                    else:
                        err_synd[synd_list].append((error.x_set,error.z_set))
                else:
                    err_synd[synd_list]= [(error.x_set,error.z_set)]
        else:
            pass

    return err_synd


def circuit_2err2(anc, extractor_z, extractor_x, two_err, one_err=None):
    err_synd = dict()
    extractor = {'X': extractor_x, 'Z': extractor_z}
    tp = 'X'
    ntp = 'XZ'.replace(tp, '')
    err_synd = add2check(extractor, err_synd, two_err, anc, tp, alph_order=True)
    err_synd = add2check(extractor, err_synd, two_err, anc, ntp, alph_order=False)
    return err_synd

def check(cir_index='c2_l1', steane=True):
    x = Check_FT(cir_index, steane)
    err = x.run()
    return err

if __name__ == '__main__':
    x = Check_FT(cir_index='c3_l2', steane=True)
    x.lut_gen()
    err = check()
    if err:
        print('Yes, this circuit is FT :)')
    else:
        print('Sorry, this circuit is not FT :(')
