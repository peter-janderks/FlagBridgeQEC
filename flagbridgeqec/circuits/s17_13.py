def cir_s17_13(section,idling):
    timesteps = []
    if section == 1:
        timesteps = cir_s17_13_first_half(timesteps,idling)
    elif section == 2:
        timesteps = cir_s17_13_second_half(timesteps,idling)
    elif section == 3:
        timesteps = cir_s17_13_second_round_after_second_half(timesteps,idling)
    elif section == 4:
        timesteps = cir_s17_13_second_round_after_first_half(timesteps,idling)
    return(timesteps)

def cir_s17_13_second_half(timesteps,idling):
    timesteps.append(op_list([('CNOT', [(80,90)]),('P',[110,130]),('H',[100,120])]))
    timesteps.append(op_list([('CNOT', [(1,80),(4,90),(100,110),(120,130)])]))
    timesteps.append(op_list([('CNOT', [(1,100),(2,80),(4,120),(5,90),(6,130),(7,110)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(4,110),(3,100),(80,90)])]))
    timesteps.append(op_list([('CNOT', [(100,110),(7,120)]),('H',[80]), ('M',[90])]))
    timesteps.append(op_list([('CNOT', [(120,130)]),('H',[100]), ('M',[80,110])]))
    timesteps.append(op_list([('H',[120]), ('M',[100,130])]))
    timesteps.append(op_list([('M',[120]),('P',[9,13])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12,13,80,90,100,110,120,130],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_s17_13_first_half(timesteps,idling):
    timesteps.append(op_list([('P', [9,13])]))
    timesteps.append(op_list([('H', [9,13]),('P', [8,11,12])]))
    timesteps.append(op_list([('CNOT', [(9,8),(13,12)]),('H',[11]),('P',[10])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,6),(8,1),(9,5),(11,10)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(11,7),(10,1),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(12,7),(10,3),(11,4),(9,8)])]))
    timesteps.append(op_list([('CNOT', [(11,10),(13,12)]),('H',[9]), ('M',[8])]))
    timesteps.append(op_list([('H', [11,13]),('M', [9,10,12]),('P', [80])]))
    timesteps.append(op_list([('M', [11,13]),('P', [90,100,120]),('H', [80])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12,13,80,90,100,110,120,130],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_s17_13_second_round_after_second_half(timesteps,idling):
    timesteps.append(op_list([('H', [9,13]),('P', [8,11,12])]))
    timesteps.append(op_list([('CNOT', [(9,8),(13,12)]),('H',[11]),('P',[10])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,6),(8,1),(9,5),(11,10)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(10,1),(11,7),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(12,7),(10,3),(11,4),(9,8)])]))
    timesteps.append(op_list([('CNOT', [(11,10),(13,12)]),('H',[9]), ('M',[8])]))
    timesteps.append(op_list([('H', [11,13]),('M', [9,10,12]),('P', [80])]))
    timesteps.append(op_list([('M', [11,13]),('P', [90,100,120]),('H', [80])]))
    timesteps.append(op_list([('CNOT', [(80,90)]),('P',[110,130]),('H',[100,120])]))
    timesteps.append(op_list([('CNOT', [(1,80),(4,90),(100,110),(120,130)])]))
    timesteps.append(op_list([('CNOT', [(1,100),(2,80),(4,120),(5,90),(6,130),(7,110)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(4,110),(3,100),(80,90)])]))
    timesteps.append(op_list([('CNOT', [(100,110),(7,120)]),('H',[80]), ('M',[90])]))
    timesteps.append(op_list([('CNOT', [(120,130)]),('H',[100]), ('M',[80,110])]))
    timesteps.append(op_list([('H',[120]), ('M',[100,130])]))
    timesteps.append(op_list([('M',[120])]))
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12,13,80,90,100,110,120,130],[1,2,3,4,5,6,7])
    return(timesteps)

def cir_s17_13_second_round_after_first_half(timesteps,idling):
    timesteps.append(op_list([('CNOT', [(80,90)]),('P',[110,130]),('H',[100,120])]))
    timesteps.append(op_list([('CNOT', [(1,80),(4,90),(100,110),(120,130)])]))
    timesteps.append(op_list([('CNOT', [(1,100),(2,80),(4,120),(5,90),(6,130),(7,110)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(4,110),(3,100),(80,90)])]))
    timesteps.append(op_list([('CNOT', [(100,110),(7,120)]),('H',[80]), ('M',[90])]))
    timesteps.append(op_list([('CNOT', [(120,130)]),('H',[100]), ('M',[80,110])]))
    timesteps.append(op_list([('H',[120]), ('M',[100,130])]))
    timesteps.append(op_list([('M',[120]),('P',[9,13])]))
    timesteps.append(op_list([('H', [9,13]),('P', [8,11,12])]))
    timesteps.append(op_list([('CNOT', [(9,8),(13,12)]),('H',[11]),('P',[10])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,6),(8,1),(9,5),(11,10)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(10,1),(11,7),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(12,7),(10,3),(11,4),(9,8)])]))
    timesteps.append(op_list([('CNOT', [(11,10),(13,12)]),('H',[9]), ('M',[8])]))
    timesteps.append(op_list([('H', [11,13]),('M', [9,10,12])]))
    timesteps.append(op_list([('M', [11,13])])) 
    if idling:
        timesteps = add_idling(timesteps, [8,9,10,11,12,13,80,90,100,110,120,130],[1,2,3,4,5,6,7])
    return(timesteps)
def cir_steane10(section):
    timesteps = []
    if section == 1:
        timesteps = cir_steane10_1st_half(timesteps)
    elif section == 2:
        timesteps = cir_steane10_2nd_half(timesteps)
    elif section == 3:
        timesteps = cir_steane10_2nd_rnd_after_1st_half(timesteps)
    elif section == 4:
        timesteps = cir_steane10_2nd_rnd_after_2nd_half(timesteps)
    return(timesteps)
    
def cir_steane10_1st_half(timesteps):
    timesteps.append(op_list([('P', [9,130])]))
    timesteps.append(op_list([('H', [9,130]),('P', [8,110,120])]))
    timesteps.append(op_list([('CNOT', [(130,120),(9,8)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(8,1),(9,5),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(9,8),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[9,130]), ('M',[8,120])]))
    timesteps.append(op_list([('H', [110]),('M', [9,100,130]),('P',[80,12])]))
    timesteps.append(op_list([('H',[80,12]),('P',[90,10,13]),('M', [110])]))
    return(timesteps)

def cir_steane10_2nd_half(timesteps):    
    timesteps.append(op_list([('CNOT', [(80,90),(12,13)]),('H',[10]),('P',[11])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,7),(1,80),(5,90),(10,11)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(13,6),(10,1),(11,7),(4,90),(2,80)])]))
    timesteps.append(op_list([('CNOT', [(10,3),(11,4),(80,90),(12,13)])]))
    timesteps.append(op_list([('CNOT', [(10,11)]),('H',[80,12]), ('M',[90,13])]))
    timesteps.append(op_list([('H', [10]),('M', [80,11,13]),('P', [9,130])]))
    timesteps.append(op_list([('M', [10]),('P', [8,110,120]),('H', [9,130])]))
    return(timesteps)

def cir_steane10_2nd_rnd_after_1st_half(timesteps):
    timesteps.append(op_list([('CNOT', [(80,90),(12,13)]),('H',[10]),('P',[11])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,7),(1,80),(5,90),(10,11)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(13,6),(10,1),(11,7),(4,90),(2,80)])]))
    timesteps.append(op_list([('CNOT', [(10,3),(11,4),(80,90),(12,13)])]))
    timesteps.append(op_list([('CNOT', [(10,11)]),('H',[80,12]), ('M',[90,13])]))
    timesteps.append(op_list([('H', [10]),('M', [80,11,13]),('P', [9,130])]))
    timesteps.append(op_list([('M', [10]),('H', [9,130]),('P', [8,110,120])]))
    timesteps.append(op_list([('CNOT', [(130,120),(9,8)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(8,1),(9,5),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(9,8),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[9,130]), ('M',[8,120])]))
    timesteps.append(op_list([('H', [110]),('M', [9,100,130])]))
    timesteps.append(op_list([('M', [110])]))
    return(timesteps)

def cir_steane10_2nd_rnd_after_2nd_half(timesteps):
    timesteps.append(op_list([('CNOT', [(130,120),(90,80)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(80,1),(90,5),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(90,4),(80,2)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(90,80),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[90,130]), ('M',[80,120])]))
    timesteps.append(op_list([('H', [110]),('M', [9,100,130]),('P',[80,12])]))
    timesteps.append(op_list([('H',[80,12]),('P',[90,10,13]),('M', [110])]))
    timesteps.append(op_list([('CNOT', [(80,90),(12,13)]),('H',[10]),('P',[11])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,7),(1,80),(5,90),(10,11)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(13,6),(10,1),(11,7),(4,90),(2,80)])]))
    timesteps.append(op_list([('CNOT', [(10,3),(11,4),(80,90),(12,13)])]))
    timesteps.append(op_list([('CNOT', [(10,11)]),('H',[80,12]), ('M',[90,13])]))
    timesteps.append(op_list([('H', [10]),('M', [80,11,13])]))
    timesteps.append(op_list([('M', [10])]))
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

