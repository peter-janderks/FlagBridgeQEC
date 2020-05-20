from flagbridgeqec.utils.circuit_construction import op_list, add_idling

def cir_steane_5a(section,idling):
    timesteps = []
    if section == 1:
        timesteps = cir_5a_first_half(timesteps,idling)
    elif section == 2:
        timesteps = cir_5a_second_half(timesteps,idling)
    elif section == 3:
        timesteps = cir_5a_second_round_after_first_half(timesteps,idling) 
    elif section == 4:
        timesteps = cir_5a_second_round_after_second_half(timesteps,idling)
    return(timesteps)

def cir_5a_first_half(timesteps,idling):
    timesteps.append(op_list([('P', [8,12])]))
    timesteps.append(op_list([('H', [8,12]),('P', [9,10,11])]))
    timesteps.append(op_list([('CNOT', [(8,9),(12,11)]),('H',[10])]))
    timesteps.append(op_list([('CNOT', [(8,4),(10,9),(11,5),(12,7)])]))

    timesteps.append(op_list([('CNOT',[(12,6),(11,4),(9,1),(8,3),(10,2)])]))
    timesteps.append(op_list([('CNOT',[(8,7),(10,4),(12,11)])]))
    
    timesteps.append(op_list([('CNOT',[(8,9),(10,5)]),('H',[(12)]),('M',[11])]))
    timesteps.append(op_list([('CNOT',[(10,9)]),('H',[8]),('M',[12]),('P',[110])]))
    
    timesteps.append(op_list([('M',[8,9]),('H',[10,110]),('P',[120])]))
    timesteps.append(op_list([('M',[10]),('CNOT',[(110,120)]),('P',[90])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_5a_second_half(timesteps,idling):
    timesteps.append(op_list([('H', [90]),('P', [80]),('CNOT',[(5,110),(7,120)])]))
    timesteps.append(op_list([('CNOT', [(90,80),(6,110),(4,120)]),('P',[100])]))
    timesteps.append(op_list([('CNOT', [(90,100),(4,80),(110,120)])]))

    timesteps.append(op_list([('M',[120]),('H',[110]),('CNOT',[(3,80),(1,90),(2,100)])]))
    timesteps.append(op_list([('CNOT',[(7,80),(4,100)]),('M',[(110)])]))
    
    timesteps.append(op_list([('CNOT',[(90,80),(5,100)])]))
    timesteps.append(op_list([('M',[80]),('CNOT',[(90,100)])]))
    timesteps.append(op_list([('H',[90]),('M',[100])]))
    timesteps.append(op_list([('M',[90]),('P',[8,12])]))
    if idling:
        timesteps = add_idling(timesteps, [80,90,100,110,120],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_5a_second_round_after_second_half(timesteps,idling):
    timesteps.append(op_list([('H', [8,12]),('P', [9,10,11])]))
    timesteps.append(op_list([('CNOT', [(8,9),(12,11)]),('H',[10])]))
    timesteps.append(op_list([('CNOT', [(8,4),(10,9),(11,5),(12,7)])]))

    timesteps.append(op_list([('CNOT',[(12,6),(11,4),(9,1),(8,3),(10,2)])]))
    timesteps.append(op_list([('CNOT',[(8,7),(10,4),(12,11)])]))

    timesteps.append(op_list([('CNOT',[(8,9),(10,5)]),('H',[(12)]),('M',[11])]))
    timesteps.append(op_list([('CNOT',[(10,9)]),('H',[8]),('M',[12]),('P',[110])]))

    timesteps.append(op_list([('M',[8,9]),('H',[10,110]),('P',[120])]))
    timesteps.append(op_list([('M',[10]),('CNOT',[(110,120)]),('P',[90])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12],[1,2,3,4,5,6,7])

    timesteps.append(op_list([('H', [90]),('P', [80]),('CNOT',[(5,110),(7,120)])]))
    timesteps.append(op_list([('CNOT', [(90,80),(6,110),(4,120)]),('P',[100])]))
    timesteps.append(op_list([('CNOT', [(90,100),(4,80),(110,120)])]))

    timesteps.append(op_list([('M',[120]),('H',[110]),('CNOT',[(3,80),(1,90),(2,100)])]))
    timesteps.append(op_list([('CNOT',[(7,80),(4,100)]),('M',[(110)])]))

    timesteps.append(op_list([('CNOT',[(90,80),(5,100)])]))
    timesteps.append(op_list([('M',[80]),('CNOT',[(90,100)])]))
    timesteps.append(op_list([('H',[90]),('M',[100])]))
    timesteps.append(op_list([('M',[90])]))
    if idling:
        timesteps = add_idling(timesteps, [80,90,100,110,120],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_5a_second_round_after_first_half(timesteps,idling):
    timesteps.append(op_list([('H', [90]),('P', [80]),('CNOT',[(5,110),(7,120)])]))
    timesteps.append(op_list([('CNOT', [(90,80),(6,110),(4,120)]),('P',[100])]))
    timesteps.append(op_list([('CNOT', [(90,100),(4,80),(110,120)])]))

    timesteps.append(op_list([('M',[120]),('H',[110]),('CNOT',[(3,80),(1,90),(2,100)])]))
    timesteps.append(op_list([('CNOT',[(7,80),(4,100)]),('M',[(110)])]))

    timesteps.append(op_list([('CNOT',[(90,80),(5,100)])]))
    timesteps.append(op_list([('M',[80]),('CNOT',[(90,100)])]))
    timesteps.append(op_list([('H',[90]),('M',[100])]))
    timesteps.append(op_list([('M',[90]),('P',[8,12])]))
    if idling:
        timesteps = add_idling(timesteps, [80,90,100,110,120],[1,2,3,4,5,6,7])
    timesteps.append(op_list([('H', [8,12]),('P', [9,10,11])]))


    timesteps.append(op_list([('CNOT', [(8,9),(12,11)]),('H',[10])]))
    timesteps.append(op_list([('CNOT', [(8,4),(10,9),(11,5),(12,7)])]))

    timesteps.append(op_list([('CNOT',[(12,6),(11,4),(9,1),(8,3),(10,2)])]))
    timesteps.append(op_list([('CNOT',[(8,7),(10,4),(12,11)])]))
                    
    timesteps.append(op_list([('CNOT',[(8,9),(10,5)]),('H',[(12)]),('M',[11])]))
    timesteps.append(op_list([('CNOT',[(10,9)]),('H',[8]),('M',[12])]))

    timesteps.append(op_list([('M',[8,9]),('H',[10])]))
    timesteps.append(op_list([('M',[10])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12],[1,2,3,4,5,6,7])
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

if __name__ == '__main__':
    print(cir_steane_5a(1,1))
    
