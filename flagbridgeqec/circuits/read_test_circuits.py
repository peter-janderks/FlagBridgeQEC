
#            translate_line(line)

#        if ' C ' in line:
#            print(line,'line')
#        elif ' H' in line:
#            print(line,'H')
def op_set_1(name, qs):
    return [(name, q) for q in qs]

def op_set_2(name, lst):
    return [(name, q[0], q[1]) for q in lst]

def support(timestep):
    """                                                                                             
    Qubits on which a list of gates act.                                                            
    """
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)

def add_idling(timestep, living_qubits):
#    dat_locs = set(anc_set + data_set)
#  for step in timesteps:
    timestep.extend([('I', q) for q in set(living_qubits) - support(timestep)])
    return timestep

def translate_line(line, living_qubits, timestep):
    if ' C ' in line:
        operations = line.split()
        timestep.append(op_set_2('CNOT',[(int(operations[0][1:]),int(operations[2][1:]))]))
    elif ' H' in line:
        operations = line.split()
        timestep.append(op_set_1('H',[int(operations[0][1:])]))
    elif 'START' in line:
        operations = line.split()
        for i in range(len(operations)-1):
            timestep.append(op_set_1('P',[int(operations[i][1:])]))
            living_qubits.extend([int(operations[i][1:])])
    
        print(timestep,'timestep')    
    return(timestep, living_qubits)
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

def compile_circuit(cir_id,idling):
    with open('test_circuit.qpic') as f:
        living_qubits = list(range(1,7))
        read_data = f.readlines()
        all_timesteps = []
        timestep = []
        living_qubits = []
        for line in read_data:
            if ' C ' in line or ' H' in line or 'START' in line:
                #            print(type(line))
                timestep, living_qubits = translate_line(line, living_qubits, timestep)
            else:
                if timestep != []:
                    add_idling(timestep, living_qubits)
                    all_timesteps.append(timestep)
                    timestep = []

if __name__ == "__main__":
    cir_id = 'test'
    idling = 1
    compile_circuit(cir_id,idling)
