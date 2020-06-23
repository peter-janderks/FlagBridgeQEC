import importlib_resources as pkg_resources
#from flagbridgeqec.circuits
import os
def op_set_1(name, qs):
    return [(name, q) for q in qs]

def op_set_2(name, lst):
    return [(name, q[0], q[1]) for q in lst]

def support(timestep):
    """                                                               
Qubits on which a list of gates act.                                          """
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)

def add_idling(timestep, living_qubits):
    timestep.extend(('I', q) for q in set(living_qubits) - support(timestep))
    return timestep

def translate_line(line, living_qubits, timestep):
    if ' C ' in line:
        operations = line.split()
        timestep.extend(op_set_2('CNOT',[(int(operations[2][1:]),int(operations[0][1:]))]))
    elif ' H' in line:
        operations = line.split()
        timestep.extend(op_set_1('H',[int(operations[0][1:])]))

    elif 'START' in line:
        operations = line.split()
        for i in range(len(operations)-1):
            timestep.extend(op_set_1('P',[int(operations[i][1:])]))
            living_qubits.extend([int(operations[i][1:])])
    elif ' M' in line:
        operations = line.split()
        timestep.extend(op_set_1('M',[int(operations[0][1:])]))
        living_qubits.remove(int(operations[0][1:]))
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

def compile_sub_circuit(cir_nm,idling,living_qubits):
    with open(cir_nm) as f:
        read_data = f.readlines()
        all_timesteps = []
        timestep = []
        
        for line in read_data:
            if ' C ' in line or ' H' in line or 'START' in line or ' M' in line:
                timestep, living_qubits = translate_line(line, living_qubits, timestep)
            elif line == '\n':
                
                if timestep != []:
                    if idling:
                        add_idling(timestep, living_qubits)
                    all_timesteps.append(timestep)
                    timestep = []

        if timestep != []:
                    if idling:
                        add_idling(timestep, living_qubits)
                    all_timesteps.append(timestep)
                    timestep = []

    return(all_timesteps,living_qubits)

def compile_circuit(cir_id,idling):
    
    living_qubits_1 = list(range(1,8))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = str(dir_path)+'/'+ str(cir_id)
#    working_directory = os.getpwd()
    first_half, living_qubits_1 = compile_sub_circuit(file_path + '/first_half.qpic',idling,living_qubits_1)
    living_qubits_2 = living_qubits_1[:]
    second_half, living_qubits_2 = compile_sub_circuit(file_path + '/second_half.qpic',idling,living_qubits_2)

    second_round_after_first_half,living_qubits_1 = compile_sub_circuit(file_path + '/second_round_after_first_half.qpic',idling,living_qubits_1)
    second_round_after_second_half,living_qubits_2 = compile_sub_circuit(file_path + '/second_round_after_second_half.qpic',idling,living_qubits_2)

    return([first_half,second_half, second_round_after_first_half, second_round_after_second_half])
if __name__ == "__main__":
    cir_id = 'test'
    idling = 1
    compile_circuit(cir_id,idling)
