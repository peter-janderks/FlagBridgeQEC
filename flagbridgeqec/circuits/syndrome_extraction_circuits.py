from flagbridgeqec.utils.error_model_2 import *
from flagbridgeqec.utils.circuit_construction import op_set_1, op_set_2, add_idling

def esmx_anc(data_set, anc_set, db=2, idling=False):
    # data_set is a list of data qubits for this x check, anc_set is a list of ancillas
    # the second ancilla is the syndrome qubit and the first is the flag qubit
    timesteps = []
    if len(anc_set) == 1:
        timesteps.append(op_set_1('P', [anc_set[0]]))
        timesteps.append(op_set_1('H', [anc_set[0]]))
        for dq in data_set:
            timesteps.append(op_set_2('CNOT', [(anc_set[0], dq)]))
        timesteps.append(op_set_1('H', [anc_set[0]]))
        timesteps.append(op_set_1('M', [anc_set[0]]))

    elif len(anc_set) == 2:
        timesteps.append(op_set_1('P', anc_set))
        timesteps.append(op_set_1('H', [anc_set[1]]))
        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))

        # Distribute a weight-x check into weight-(a+b) a=db
        t1 = []
        t2 = []
        for dq in data_set[:db]:
            t1.append(op_set_2('CNOT', [(anc_set[0], dq)]))
            # timesteps.append(op_set_2('CNOT', [(anc_set[0], dq)]))
        for dq in data_set[db:]:
            t2.append(op_set_2('CNOT', [(anc_set[1], dq)]))
            # timesteps.append(op_set_2('CNOT', [(anc_set[1], dq)]))
        if len(t1) >= len(t2):
            tl = t1
            ts = t2
        else:
            tl = t2
            ts = t1

        for i in range(len(ts)):
            timesteps.append(tl[i]+ts[i])
        for i in range(len(ts), len(tl)):
            timesteps.append(tl[i])

        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))
        timesteps.append(op_set_1('H', [anc_set[1]]))
        timesteps.append(op_set_1('M', anc_set))
        
    elif len(anc_set) == 3:
        timesteps.append(op_set_1('P', anc_set))
        timesteps.append(op_set_1('H', [anc_set[0]]))
        timesteps.append(op_set_2('CNOT', [(anc_set[0], anc_set[1])]))
        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[2])]))

        # Distribute a weight-z check into weight-(a+b+c)
        # tobefixed for parallelism
        timesteps.append(op_set_2('CNOT', [(anc_set[0], data_set[0]), (anc_set[1], data_set[2]), (anc_set[2], data_set[3])]))
        timesteps.append(op_set_2('CNOT', [(anc_set[0], data_set[1])]))

        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[2])]))
        timesteps.append(op_set_2('CNOT', [(anc_set[0], anc_set[1])]))
        timesteps.append(op_set_1('H', [anc_set[0]]))
        timesteps.append(op_set_1('M', anc_set))

    else:
        raise('Only support one or two ancillas for parity checks')
    if idling:
        timesteps = add_idling(timesteps, anc_set, data_set=[1,2,3,4,5,6,7])
    return(timesteps)

def esmz_anc(data_set, anc_set, db=2, cd=3, idling=False):
    # data_set is a list of data qubits for this z check, anc_set is a list of ancillas
    # the first ancilla is the syndrome qubit and the second is the flag qubit
    timesteps = []
    if len(anc_set) == 1:
        timesteps.append(op_set_1('P', [anc_set[0]]))
        for dq in data_set:
            timesteps.append(op_set_2('CNOT', [(dq, anc_set[0])]))
        timesteps.append(op_set_1('M', [anc_set[0]]))

    elif len(anc_set) == 2:
        timesteps.append(op_set_1('P', anc_set))
        timesteps.append(op_set_1('H', [anc_set[1]]))
        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))

        # Distribute a weight-z check into weight-(a+b) a = db
        t1 = []
        t2 = []
        for dq in data_set[:db]:
            t1.append(op_set_2('CNOT', [(dq, anc_set[0])]))
            # timesteps.append(op_set_2('CNOT', [(dq, anc_set[0])]))
        for dq in data_set[db:]:
            t2.append(op_set_2('CNOT', [(dq, anc_set[1])]))
            # timesteps.append(op_set_2('CNOT', [(dq, anc_set[1])]))
        if len(t1) >= len(t2):
            tl = t1
            ts = t2
        else:
            tl = t2
            ts = t1

        for i in range(len(ts)):
            timesteps.append(tl[i]+ts[i])
        for i in range(len(ts), len(tl)):
            timesteps.append(tl[i])

        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))
        timesteps.append(op_set_1('H', [anc_set[1]]))
        timesteps.append(op_set_1('M', anc_set))

    elif len(anc_set) == 3:
        timesteps.append(op_set_1('P', anc_set))
        timesteps.append(op_set_1('H', anc_set[1:]))
        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))
        timesteps.append(op_set_2('CNOT', [(anc_set[2], anc_set[1])]))

        # Distribute a weight-z check into weight-(a+b+c)
        # tobefixed for parallelism
        timesteps.append(op_set_2('CNOT', [(data_set[0], anc_set[0]), (data_set[2], anc_set[1]), (data_set[3], anc_set[2])]))
        timesteps.append(op_set_2('CNOT', [(data_set[1], anc_set[0])]))
        # for dq in data_set[:2]:
        #     timesteps.append(op_set_2('CNOT', [(dq, anc_set[0])]))
        # for dq in data_set[2:3]:
        #     timesteps.append(op_set_2('CNOT', [(dq, anc_set[1])]))
        # for dq in data_set[3:]:
        #     timesteps.append(op_set_2('CNOT', [(dq, anc_set[2])]))
        timesteps.append(op_set_2('CNOT', [(anc_set[2], anc_set[1])]))
        timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))
        timesteps.append(op_set_1('H', anc_set[1:]))
        timesteps.append(op_set_1('M', anc_set))

    else:
        raise('Only support one or two ancillas for parity checks')
    
    if idling:
        timesteps = add_idling(timesteps, anc_set, data_set=[1,2,3,4,5,6,7])

    return timesteps


def esmzs_anc3(data_set1, data_set2, anc_set1, idling=False):
    # two checks in parallel and then another check
    # data_set1 = [3,7,2, 5] data_set2 = [3,7,1,4] 
    # anc_set1 = [8, 10, 9] 
    timesteps = []

    timesteps.append(op_set_1('P', [i for i in anc_set1]))
    timesteps.append(op_set_1('H', [anc_set1[-1]]))

    timesteps.append(op_set_2('CNOT', [(anc_set1[-1], anc_set1[0])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[-1], anc_set1[1])]))
    timesteps.append(op_set_2('CNOT', [(data_set1[2], anc_set1[0]), (data_set2[0], anc_set1[1]), (data_set2[2], anc_set1[2])]))
    timesteps.append(op_set_2('CNOT', [(data_set1[3], anc_set1[0]), (data_set2[1], anc_set1[1]), (data_set2[3], anc_set1[2])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[-1], anc_set1[1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[-1], anc_set1[0])]))
    
    timesteps.append(op_set_1('H', [anc_set1[-1] ]))
    timesteps.append(op_set_1('M', [i for i in anc_set1]))
    
    if idling:
        timesteps = add_idling(timesteps, anc_set1, [1,2,3,4,5,6,7])
    return timesteps

def esmxs_anc3(data_set1, data_set2, anc_set1, chao=False, idling=False):

    timesteps = []

    timesteps.append(op_set_1('P', [i for i in anc_set1]))
    timesteps.append(op_set_1('H', [i for i in anc_set1[:2]]))

    timesteps.append(op_set_2('CNOT', [(anc_set1[0], anc_set1[-1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[1], anc_set1[-1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[0], data_set1[2]), (anc_set1[1], data_set2[0]), (anc_set1[2], data_set2[2])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[0], data_set1[3]), (anc_set1[1], data_set2[1]), (anc_set1[2], data_set2[3])]))
    
    timesteps.append(op_set_2('CNOT', [(anc_set1[0], anc_set1[-1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set1[1], anc_set1[-1])]))

    timesteps.append(op_set_1('H', [i for i in anc_set1[:2]]))
    timesteps.append(op_set_1('M', [i for i in anc_set1]))

    
    if idling:
        timesteps = add_idling(timesteps, anc_set1, [1,2,3,4,5,6,7])
    return timesteps

def esmzs_anc4(data_set1, data_set2, data_set3, anc_set, chao=False, idling=False):
    #  three checks in parallel 
    # data_set1 = [4,1,5,2] data_set2 = [4,1,3,7] data_set3 = [4,5,6,7] 
    # anc_set = [8, 10, 12, 9]                                                 
    timesteps = []

    timesteps.append(op_set_1('P', [i for i in anc_set]))
    timesteps.append(op_set_1('H', [anc_set[-1]]))

    timesteps.append(op_set_2('CNOT', [(anc_set[3], anc_set[2])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[2], anc_set[1]), (data_set3[2], anc_set[3])]))
    timesteps.append(op_set_2('CNOT', [(data_set2[2], anc_set[1]), (data_set3[1], anc_set[3]), (data_set2[3], anc_set[2])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0]), (data_set3[0], anc_set[3])]))
    timesteps.append(op_set_2('CNOT', [(data_set1[3], anc_set[0]), (data_set2[1], anc_set[1])]))
    timesteps.append(op_set_2('CNOT', [(data_set1[2], anc_set[0]), (data_set2[0], anc_set[1])]))

    timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[0])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[2], anc_set[1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[3], anc_set[2])]))

    timesteps.append(op_set_1('H', [anc_set[-1] ]))
    timesteps.append(op_set_1('M', [i for i in anc_set]))

    
    if idling:
        timesteps = add_idling(timesteps, anc_set, [1,2,3,4,5,6,7])
    return timesteps

def esmxs_anc4(data_set1, data_set2, data_set3, anc_set, chao=False, idling=False):
    #  three checks in parallel 
    # data_set1 = [4,1,5,2] data_set2 = [4,1,3,7] data_set3 = [4,5,6,7] 
    # anc_set = [80, 100, 120, 90] 
    timesteps = []

    timesteps.append(op_set_1('P', [i for i in anc_set]))
    timesteps.append(op_set_1('H', anc_set[:3]))
    
    timesteps.append(op_set_2('CNOT', [(anc_set[2], anc_set[3])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[2]), (anc_set[3], data_set3[2])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[1], data_set2[2]), (anc_set[3], data_set3[1]), (anc_set[2], data_set2[3])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[0], anc_set[1]), (anc_set[3], data_set3[0])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[0], data_set1[3]), (anc_set[1], data_set2[1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[0], data_set1[2]), (anc_set[1], data_set2[0])]))

    timesteps.append(op_set_2('CNOT', [(anc_set[0], anc_set[1])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[1], anc_set[2])]))
    timesteps.append(op_set_2('CNOT', [(anc_set[2], anc_set[3])]))


    timesteps.append(op_set_1('H', anc_set[:3]))
    timesteps.append(op_set_1('M', [i for i in anc_set]))

    
    if idling:
        timesteps = add_idling(timesteps, anc_set, [1,2,3,4,5,6,7])
    return timesteps

