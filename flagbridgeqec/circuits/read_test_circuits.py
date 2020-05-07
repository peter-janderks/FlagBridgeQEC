with open('test_circuit.qpic') as f:
    read_data = f.read()
    for line in read_data:
        print(read_data)

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
