def op_list(ls_ops):
    operators = []
    for i in range(len(ls_ops)):
        if type(ls_ops[i][1][0]) == int:
            for j in ls_ops[i][1]:
                operators.append((ls_ops[i][0],j))
        else:
            for j in ls_ops[i][1]:
                operators.append((ls_ops[i][0],j[0],j[1]))
    return(operators)

def op_set_2(name, lst):
    return [(name, q[0], q[1]) for q in lst]


def op_set_1(name, qs):
    return [(name, q) for q in qs]

def add_idling(timesteps, anc_set, data_set):
    dat_locs = set(anc_set + data_set)
    for step in timesteps:
        step.extend([('I', q) for q in dat_locs - support(step)])
    return timesteps

def support(timestep):
    """               
    Qubits on which a list of gates act.                                          
    """
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)
