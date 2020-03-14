def cir_steane_6a_ft(section):
    """                                                                                  New circuit to perform each check for the Steane code using two ancillas per face    """

    timesteps = []
    if section == 2 or section == 3:

        timesteps.append(op_set_1('P', [8,9,10,11,12,13]))
        timesteps.append(op_set_1('H', [8,10,12]))
        timesteps.append(op_set_2('CNOT', [(12,13), (10,11),(8,9)]))

        timesteps.append(op_set_2('CNOT', [(12,4),(13,7),(8,1),(9,5)]))
        timesteps.append(op_set_2('CNOT', [(12,5),(13,6),(10,1),(11,7),(9,4),(8,2)]))
        timesteps.append(op_set_2('CNOT', [(10,3),(11,4)]))

        timesteps.append(op_set_2('CNOT', [(12,13), (10,11),(8,9)]))
        timesteps.append(op_set_1('H', [8,10,12]))
        timesteps.append(op_set_1('M', [8,9,10,11,12,13]))

    if section == 1 or section == 3:

        timesteps.append(op_set_1('P', [80,90,100,110,120,130]))
        timesteps.append(op_set_1('H', [90,110,130]))
        timesteps.append(op_set_2('CNOT', [(130,120), (110,100),(90,80)]))
        timesteps.append(op_set_2('CNOT', [(4,120),(7,130),(1,80),(5,90)]))
        timesteps.append(op_set_2('CNOT', [(5,120),(6,130),(1,100),(7,110),(4,90),(2,80)]))
        timesteps.append(op_set_2('CNOT', [(4,110),(3,100)]))
        timesteps.append(op_set_2('CNOT', [(130,120), (110,100),(90,80)]))
        timesteps.append(op_set_1('H', [90,110,130]))
        timesteps.append(op_set_1('M', [80,90,100,110,120,130]))

    return timesteps

#def cir_steane_6a(section):
#    """                                         
#    New circuit to perform each check for the Steane code using two ancillas per face         """
#    timesteps = []
#    if section == 1 or section == 3:
#        timesteps.append(op_set_1('P', [8,9,100,110,120,130]))
#        timesteps.append(op_set_1('H', [9,110,130]))
#        timesteps.append(op_set_2('CNOT', [(130,120), (110,100),(9,8)]))

#        timesteps.append(op_set_2('CNOT', [(4,120),(7,130),(8,1),(9,5)]))
#        timesteps.append(op_set_2('CNOT', [(5,120),(6,130),(1,100),(7,110),(9,4),(8,2)]))
#        timesteps.append(op_set_2('CNOT', [(3,100),(4,110)]))

#        timesteps.append(op_set_2('CNOT', [(130,120), (110,100),(9,8)]))
#        timesteps.append(op_set_1('H', [9,110,130]))
#        timesteps.append(op_set_1('M', [8,9,100,110,120,130]))

#    if section == 2 or section ==3:
#        timesteps.append(op_set_1('P', [80,90,10,11,12,13]))
#        timesteps.append(op_set_1('H', [90,11,13]))
#        timesteps.append(op_set_2('CNOT', [(13,12), (11,10),(90,80)]))

#        timesteps.append(op_set_2('CNOT', [(12,4),(13,7),(1,80),(5,90)]))
#        timesteps.append(op_set_2('CNOT', [(12,5),(13,6),(10,1),(11,7),(4,90),(2,80)]))
#        timesteps.append(op_set_2('CNOT', [(11,4),(10,3)]))

#        timesteps.append(op_set_2('CNOT', [(13,12), (11,10),(90,80)]))
#        timesteps.append(op_set_1('H', [90,11,13]))
#        timesteps.append(op_set_1('M', [80,90,10,11,12,13]))
#    return timesteps

def cir_steane_6a(section):
    timesteps = []
    if section == 1:
        timesteps = cir_6a_first_half(timesteps)
    elif section == 2:
        timesteps = cir_6a_second_half(timesteps)
    elif section == 3:
        timesteps = cir_6a_second_round_after_second_half(timesteps)
    elif section == 4:
        timesteps = cir_6a_second_round_after_first_half(timesteps)
    return(timesteps)

def cir_6a_second_half(timesteps):
    timesteps.append(op_list([('CNOT', [(130,120),(90,80)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(1,80),(5,90),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(4,90),(2,80)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(90,80),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[90,130]), ('M',[80,120])]))
    timesteps.append(op_list([('H', [110]),('M', [90,100,130]),('P',[8,12])]))
    timesteps.append(op_list([('H',[8,12]),('P',[9,10,13]),('M', [110])]))
    return(timesteps)

def cir_6a_first_half(timesteps):
    timesteps.append(op_list([('P', [8,12])]))
    timesteps.append(op_list([('H', [8,12]),('P', [9,10,13])]))
    timesteps.append(op_list([('CNOT', [(8,9),(12,13)]),('H',[10]),('P',[11])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,7),(8,1),(9,5),(10,11)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(13,6),(10,1),(11,7),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(10,3),(11,4),(8,9),(12,13)])]))
    timesteps.append(op_list([('CNOT', [(10,11)]),('H',[8,12]), ('M',[9,13])]))
    timesteps.append(op_list([('H', [10]),('M', [8,11,12]),('P', [90,130])]))
    timesteps.append(op_list([('M', [10]),('P', [80,110,120]),('H', [90,130])]))
    return(timesteps)

def cir_6a_second_round_after_second_half(timesteps):
    timesteps.append(op_list([('CNOT', [(8,9),(12,13)]),('H',[10]),('P',[11])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,7),(8,1),(9,5),(10,11)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(13,6),(10,1),(11,7),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(10,3),(11,4),(8,9),(12,13)])]))
    timesteps.append(op_list([('CNOT', [(10,11)]),('H',[8,12]), ('M',[9,13])]))
    timesteps.append(op_list([('H', [10]),('M', [8,11,12]),('P', [90,130])]))
    timesteps.append(op_list([('M', [10]),('H', [90,130]),('P', [80,110,120])]))
    timesteps.append(op_list([('CNOT', [(130,120),(90,80)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(1,80),(5,90),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(4,90),(2,80)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(90,80),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[90,130]), ('M',[80,120])]))
    timesteps.append(op_list([('H', [110]),('M', [90,100,130])]))
    timesteps.append(op_list([('M', [110])]))
    return(timesteps)

def cir_6a_second_round_after_first_half(timesteps):
    timesteps.append(op_list([('CNOT', [(130,120),(90,80)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(1,80),(5,90),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(4,90),(2,80)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(90,80),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[90,130]), ('M',[80,120])]))
    timesteps.append(op_list([('H', [110]),('M', [90,100,130]),('P',[8,12])]))
    timesteps.append(op_list([('H',[8,12]),('P',[9,10,13]),('M', [110])]))
    timesteps.append(op_list([('CNOT', [(8,9),(12,13)]),('H',[10]),('P',[11])]))
    timesteps.append(op_list([('CNOT', [(12,4),(13,7),(8,1),(9,5),(10,11)])]))
    timesteps.append(op_list([('CNOT', [(12,5),(13,6),(10,1),(11,7),(9,4),(8,2)])]\
))
    timesteps.append(op_list([('CNOT', [(10,3),(11,4),(8,9),(12,13)])]))
    timesteps.append(op_list([('CNOT', [(10,11)]),('H',[8,12]), ('M',[9,13])]))
    timesteps.append(op_list([('H', [10]),('M', [8,11,12])]))
    timesteps.append(op_list([('M', [10])]))
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
    timesteps.append(op_list([('CNOT', [(130,120),(9,8)]),('P',[100]),('H',[110])]))
    timesteps.append(op_list([('CNOT', [(4,120),(7,130),(8,1),(9,5),(110,100)])]))
    timesteps.append(op_list([('CNOT', [(5,120),(6,130),(1,100),(7,110),(9,4),(8,2)])]))
    timesteps.append(op_list([('CNOT', [(3,100),(4,110),(9,8),(130,120)])]))
    timesteps.append(op_list([('CNOT', [(110,100)]),('H',[9,130]), ('M',[8,120])]))
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

