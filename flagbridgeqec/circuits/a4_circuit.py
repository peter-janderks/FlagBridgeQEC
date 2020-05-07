from flagbridgeqec.utils.circuit_construction import op_set_1, op_set_2, add_idling

def cir_steane_4a(section,idling):
    timesteps = []
    if section == 1:
        timesteps = cir_4a_first_half(timesteps,idling)
    elif section == 2:
        timesteps = cir_4a_second_half(timesteps,idling)
    elif section == 3:
        timesteps = cir_4a_second_round_after_first_half(timesteps,idling) 
    elif section == 4:
        timesteps = cir_4a_second_round_after_second_half(timesteps,idling)
    return(timesteps)

def cir_4a_first_half(timesteps,idling):
    timesteps.append(op_set_1('P', [11]))
    timesteps.append(op_list([('H', [11]),('P', [10])]))
    timesteps.append(op_list([('CNOT', [(11,10)]),('P',[9])]))
    timesteps.append(op_list([('CNOT', [(10,9),(6,11)])]))
    timesteps.append(op_list([('P',[8]),('CNOT',[(3,9),(5,11),(7,10)])]))
    timesteps.append(op_list([('CNOT',[(9,8),(4,11)])]))
    timesteps.append(op_list([('CNOT',[(2,8),(1,9)])]))
    timesteps.append(op_list([('CNOT',[(5,8),(4,9)])]))
    timesteps.append(op_list([('CNOT',[(9,8)])]))
    timesteps.append(op_list([('M',[8]),('CNOT',[(10,9)])]))
    timesteps.append(op_list([('M',[9]),('CNOT',[(11,10)])]))
    timesteps.append(op_list([('M',[10]),('H',[11])]))
    timesteps.append(op_list([('M',[11]),('P',[100])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,80,90,100,110],[1,2,3,4,5,6,7])
    return(timesteps)



def cir_4a_second_half(timesteps,idling):
    timesteps.append(op_list([('H', [100]),('P', [110,90])]))
    timesteps.append(op_list([('CNOT', [(100,110)]),('H',[90])]))
    timesteps.append(op_list([('CNOT', [(90,100),(110,6)]),('P',[80])]))
    timesteps.append(op_list([('H',[80]),('CNOT',[(90,3),(110,5),(100,7)])]))
    timesteps.append(op_list([('CNOT',[(80,90),(110,4)])]))
    timesteps.append(op_list([('CNOT',[(80,2),(90,1)])]))
    timesteps.append(op_list([('CNOT',[(80,5),(90,4)])]))
    timesteps.append(op_list([('CNOT',[(80,90)])]))
    timesteps.append(op_list([('H',[80]),('CNOT',[(90,100)])]))
    timesteps.append(op_list([('H',[90]),('CNOT',[(100,110)]),('M',[80])]))
    timesteps.append(op_list([('M',[90,110]),('H',[100])]))
    timesteps.append(op_list([('M',[100]),('P',[11])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,80,90,100,110],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_4a_second_round_after_first_half(timesteps,idling):
    timesteps.append(op_list([('H', [100]),('P', [110,90])]))
    timesteps.append(op_list([('CNOT', [(100,110)]),('H',[90])]))
    timesteps.append(op_list([('CNOT', [(90,100),(110,6)]),('P',[80])]))
    timesteps.append(op_list([('H',[80]),('CNOT',[(90,3),(110,5),(100,7)])]))
    timesteps.append(op_list([('CNOT',[(80,90),(110,4)])]))
    timesteps.append(op_list([('CNOT',[(80,2),(90,1)])]))
    timesteps.append(op_list([('CNOT',[(80,5),(90,4)])]))
    timesteps.append(op_list([('CNOT',[(80,90)])]))
    timesteps.append(op_list([('H',[80]),('CNOT',[(90,100)])]))
    timesteps.append(op_list([('H',[90]),('CNOT',[(100,110)]),('M',[80])]))
    timesteps.append(op_list([('M',[90,110]),('H',[100])]))
    timesteps.append(op_list([('M',[100]),('P',[11])]))
    timesteps.append(op_list([('H', [11]),('P', [10])]))
    timesteps.append(op_list([('CNOT', [(11,10)]),('P',[9])]))
    timesteps.append(op_list([('CNOT', [(10,9),(6,11)])]))
    timesteps.append(op_list([('P',[8]),('CNOT',[(3,9),(5,11),(7,10)])]))
    timesteps.append(op_list([('CNOT',[(9,8),(4,11)])]))
    timesteps.append(op_list([('CNOT',[(2,8),(1,9)])]))
    timesteps.append(op_list([('CNOT',[(5,8),(4,9)])]))
    timesteps.append(op_list([('CNOT',[(9,8)])]))
    timesteps.append(op_list([('M',[8]),('CNOT',[(10,9)])]))
    timesteps.append(op_list([('M',[9]),('CNOT',[(11,10)])]))
    timesteps.append(op_list([('M',[10]),('H',[11])]))
    timesteps.append(op_list([('M',[11])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,80,90,100,110],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_4a_second_round_after_second_half(timesteps,idling):
    timesteps.append(op_list([('H', [11]),('P', [10])]))
    timesteps.append(op_list([('CNOT', [(11,10)]),('P',[9])]))
    timesteps.append(op_list([('CNOT', [(10,9),(6,11)])]))
    timesteps.append(op_list([('P',[8]),('CNOT',[(3,9),(5,11),(7,10)])]))
    timesteps.append(op_list([('CNOT',[(9,8),(4,11)])]))
    timesteps.append(op_list([('CNOT',[(2,8),(1,9)])]))
    timesteps.append(op_list([('CNOT',[(5,8),(4,9)])]))
    timesteps.append(op_list([('CNOT',[(9,8)])]))
    timesteps.append(op_list([('M',[8]),('CNOT',[(10,9)])]))
    timesteps.append(op_list([('M',[9]),('CNOT',[(11,10)])]))
    timesteps.append(op_list([('M',[10]),('H',[11])]))
    timesteps.append(op_list([('M',[11]),('P',[100])]))
    timesteps.append(op_list([('H', [100]),('P', [110,90])]))
    timesteps.append(op_list([('CNOT', [(100,110)]),('H',[90])]))
    timesteps.append(op_list([('CNOT', [(90,100),(110,6)]),('P',[80])]))
    timesteps.append(op_list([('H',[80]),('CNOT',[(90,3),(110,5),(100,7)])]))
    timesteps.append(op_list([('CNOT',[(80,90),(110,4)])]))
    timesteps.append(op_list([('CNOT',[(80,2),(90,1)])]))
    timesteps.append(op_list([('CNOT',[(80,5),(90,4)])]))
    timesteps.append(op_list([('CNOT',[(80,90)])]))
    timesteps.append(op_list([('H',[80]),('CNOT',[(90,100)])]))
    timesteps.append(op_list([('H',[90]),('CNOT',[(100,110)]),('M',[80])]))
    timesteps.append(op_list([('M',[90,110]),('H',[100])]))
    timesteps.append(op_list([('M',[100])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,80,90,100,110],[1,2,3,4,5,6,7])
    return(timesteps)


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

def op_set_1(name, qs):
    return [(name, q) for q in qs]

def op_set_2(name, lst):
    return [(name, q[0], q[1]) for q in lst]
