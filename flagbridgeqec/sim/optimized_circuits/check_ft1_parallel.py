import circuit_metric as cm
import sparse_pauli as sp

from flagbridgeqec.circuits import cir_steane_4a, cir_steane_5a, cir_steane_6a, cir_s17_13
from flagbridgeqec.circuits.read_test_circuits import compile_circuit

flatten = lambda l: [item for sublist in l for item in sublist]

class Check_FT(object):
    def __init__(self, cir_index=2):

        self.stabilisers = {'X': {8: sp.Pauli(x_set=[1,2,4,5]),
                                  10: sp.Pauli(x_set=[1,3,4,7]),
                                  12:  sp.Pauli(x_set=[4,5,6,7])},
                            'Z': {80: sp.Pauli(z_set=[1,2,4,5]),
                                  100: sp.Pauli(z_set=[1,3,4,7]),
                                  120:  sp.Pauli(z_set=[4,5,6,7])}}

        if cir_index == 'IBM_11':
            self.q_synd = [8,9,10,80,90,100]
            self.q_flag = [11, 110]
            self.q_syndx = [80, 90, 100]
            self.q_syndz = [8, 9, 10]

        elif cir_index == 'IBM_12':
            self.q_synd = [8,10,12,80,100,120]
            self.q_flag = [9,11,90,110]
            self.q_syndx = [80,100,120]
            self.q_syndz = [8,10,12]

        elif cir_index == 'IBM_13':
            self.q_flag = [8,10,12,80,100,120]
            self.q_synd = [130, 110, 90, 9, 11, 13]
            self.q_syndx = [130,110,90]
            self.q_syndz = [9, 11, 13]

        elif cir_index == 's17_33':
            self.q_synd = [8,10,11, 80,100,110]
            self.q_flag = [90, 120, 130, 9, 12, 13]
            self.q_syndx = [80, 100, 110]
            self.q_syndz = [8, 10, 11]

        elif cir_index == 's17_222':
            self.q_synd = [9,11,13,90,110,130]
            self.q_flag = [80, 100, 120, 8, 10, 12]
            self.q_syndx = [90, 110, 130]
            self.q_syndz = [9, 11, 13]

        elif cir_index == 's17_13':
            self.q_synd = [9,11,13,90,110,130]
            self.q_flag = [80, 100, 120, 8, 10, 12]
            self.q_syndx = [9, 11, 13]
            self.q_syndz = [90, 110, 130]

        self.cir_index = cir_index

        self.ancillas = self.q_synd + self.q_flag

        self.init_stabs = [sp.Pauli(x_set=[1,2,4,5]), 
                           sp.Pauli(x_set=[4,5,6,7]), 
                           sp.Pauli(x_set=[1,3,4,7]), 
                           sp.Pauli(z_set=[1,2,4,5]), 
                           sp.Pauli(z_set=[4,5,6,7]), 
                           sp.Pauli(z_set=[1,3,4,7])]
        
        self.init_logs = [sp.Pauli(x_set=[1,2,3,4,5,6,7]),
                          sp.Pauli(z_set=[1,2,3,4,5,6,7])]

                
        
        self.two_error = ['IX', 'IZ', 'IY', 'XI', 'ZI', 'YI', 'XX', 'XZ', 
                          'XY','ZX', 'ZZ', 'ZY', 'YX', 'YZ', 'YY']
        self.one_error = ['IX', 'IZ', 'IY']

        self.flat_circuit = self.circuit_and_statistics()

    def ft_circuit(self,idling=False):
        timestps = []
        if self.cir_index == 'IBM_11':
            timestps = compile_circuit('IBM_11',idling)
            self.x_ancillas = set((80,90,100,110))
            self.z_ancillas = set((8,9,10,11))

        elif self.cir_index == 'IBM_12':
            timestps = compile_circuit('IBM_12',idling)
            self.x_ancillas = set((80,90,100,110,120))
            self.z_ancillas = set((8,9,10,11,12))

        elif self.cir_index == 'IBM_13':
            timestps = compile_circuit('IBM_13',idling)
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))

        elif self.cir_index == 's17_222':
            timestps = compile_circuit('s17_222',idling)
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))

        elif self.cir_index == 's17_33':
            timestps = compile_circuit('s17_33',idling)
            self.x_ancillas = set((80,90,100,110,120,130))
            self.z_ancillas = set((8,9,10,11,12,13))

        return timestps

    def circuit_and_statistics(self):
        four_circuit_halves = self.ft_circuit()
        full_circuit = flatten(four_circuit_halves)
        idling = 0
        first_round = flatten(four_circuit_halves[0:2])
        
        flat_first_round = flatten(first_round)
        cnot_n = 0
        cnot_nd = 0
        flat_first_round = flatten(first_round)
        print('timesteps number', len(first_round))
        print('gate number', len(flat_first_round))
        for i in flat_first_round:
            if i[0] == 'CNOT':
                cnot_n += 1
                if len(set(self.ancillas) & set(i[1:])) != 2:
                    cnot_nd += 1

        print('cnot number', cnot_n)
        print('cnot on data number', cnot_nd)
        
        return(four_circuit_halves)
        
    def lut_gen(self):
        err_synd_pairs = self.generate_err_synd_pairs()
        lut_synd = dict()
        lut_flag = dict()

        for synd in err_synd_pairs.keys():

            if len(synd[0]) + len(synd[1]) == 0:
                pass

            else:
                synd_r1 = set(synd[0]) | set(synd[1])

                synd_r2 = set(synd[2]) | set(synd[3])
                err_minx = self.init_logs[0]
                err_minz = self.init_logs[1]
                for qbt in err_synd_pairs[synd]:
                    err = sp.Pauli(x_set=qbt[0])
                    if err.weight() < err_minx.weight():
                        err_minx = err

                for qbt in err_synd_pairs[synd]:
                    err = sp.Pauli(z_set=qbt[1])
                    if err.weight() < err_minz.weight():
                        err_minz = err

                synd_r2 = [set(synd[2]),set(synd[3])]
                err_mins = [err_minx, err_minz]

                if synd_r1 & set(self.q_flag):

                    # if flags are measured in the first round
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
                                        pass
#                                        raise('Possibly not distingushed syndromes')
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
                                        pass
#                                        raise('Possibly not distingushed syndromes')

        for flag in lut_flag.keys():
            for synd in lut_synd.keys():
                if synd not in lut_flag[flag].keys():
                    lut_flag[flag][synd] = lut_synd[synd]
        

        return(lut_synd,lut_flag)


    def generate_err_synd_pairs(self):
        full_circuit = self.flat_circuit
        err_synd = {}
        f_c1 = flatten(full_circuit[0])
        f_c2 = flatten(full_circuit[2])
        err_synd = self.half_err_synd_pairs(err_synd,f_c1,f_c2,self.x_ancillas)
        f_c1 = flatten(full_circuit[1])
        f_c2 = flatten(full_circuit[3])
        err_synd = self.half_err_synd_pairs(err_synd,f_c1,f_c2,self.z_ancillas)
        return(err_synd)


    def half_err_synd_pairs(self,err_synd,f_c1,f_c2,one_type_ancillas):
        for i in range(len(f_c1)):
            
            if len(list(f_c1[i])) == 3:
            # if it is a two-qubit gate    

                for err in self.two_error:
                    # add 16 kinds of pauli errors one by one

                    error = sp.Pauli()
                    if err[0] == 'X' or err[0] == 'Y':
                        error *= sp.Pauli({int(list(f_c1[i])[1])}, {})
                    if err[0] == 'Z' or err[0] == 'Y':
                        error *= sp.Pauli({}, {int(list(f_c1[i])[1])})
                    if err[1] == 'X' or err[1] == 'Y':
                        error *= sp.Pauli({int(list(f_c1[i])[-1])}, {})
                    if err[1] == 'Z' or err[1] == 'Y':
                        error *= sp.Pauli({}, {int(list(f_c1[i])[-1])})
                        
                    syndromes1, error = esm_apply_err(f_c1, 
                                                      one_type_ancillas, 
                                                      error, 
                                                      rg=i+1, 
                                                      bk=False)
                    # The execution breaks here because we assume if we see a 
                    # syndrome or a flag, we will stop this round and start a full of
                    # new round 
                    syndromes2, error = esm_apply_err_skip_P(f_c2, self.ancillas, error)

                    syndromes1_x = (syndromes1 & self.x_ancillas)
                    syndromes1_z = (syndromes1 & self.z_ancillas)
                    syndromes2_x = (syndromes2 & self.x_ancillas)
                    syndromes2_z = (syndromes2 & self.z_ancillas)

                    synd_list = (tuple(syndromes1_x),tuple(syndromes1_z),tuple(syndromes2_x),tuple(syndromes2_z))

#                    if syndromes1_x == set() and syndromes1_z == set():
#                        pass 

                    if synd_list in err_synd.keys():
                        if (error.x_set, error.z_set) in err_synd[synd_list]:
                            pass
                        else:
                            err_synd[synd_list].append((error.x_set,error.z_set))
                    else:
                        err_synd[synd_list]= [(error.x_set,error.z_set)]
            else:
                pass

        return(err_synd)

    def run(self):
        err_synds = self.generate_err_synd_pairs()
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
                                print(err1,'err1')
                                print(err2,'err2')
                                ft_value = 0

                                print('One case that errors are going to be a logical,here')
                                print(synd_list,'synd_list')
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

def esm_apply_err(extractor, anc, errors, rg=0, bk=False):
    firing_ancilla = set()
    for j in range(rg, len(extractor)):
        if extractor[j][0] == 'P':
            if bk:
                break
        else:
            ancilla_measurements, errors = cm.apply_step([extractor[j]], errors)
            firing_ancilla |= ancilla_measurements[1]
    errors.prep(anc)
    return firing_ancilla, errors

def esm_apply_err_skip_P(extractor, anc, errors, rg=0, bk=False):
    firing_ancilla = set()
    for j in range(rg, len(extractor)):
        ancilla_measurements, errors = cm.apply_step([extractor[j]], errors)
        firing_ancilla |= ancilla_measurements[1]
    errors.prep(anc)

    return firing_ancilla, errors

if __name__ == '__main__':
    x = Check_FT(cir_index='IBM_11')
    x.lut_gen()
    err = x.run()
    if err:
         print('Yes, this circuit is FT :)')
    else:
         print('Sorry, this circuit is not FT :(')

