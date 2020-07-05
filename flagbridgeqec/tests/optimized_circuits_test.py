import pytest
from flagbridgeqec.sim.optimized_circuits.run_circuit import check
from flagbridgeqec.sim.optimized_circuits.check_ft1_parallel import Check_FT

circuits = ['s17_222','s17_33','IBM_11', 'IBM_12','IBM_13']

def test_if_circuits_are_ft():
    for cir_id in circuits:
        x = Check_FT(cir_id)
        x.lut_gen()
        err = x.run()
        assert err == 1

def test_lut_decoder_sequential_circuit():
    for cir_id in circuits:
        # run circuit 'cir_id' 100 times for p1=p2=pm (probability of pauli error on 1 qubit gate,
        # 2 qubit gate and prob. of spam error and pI=0 (prob. of pauli error on idling gate) 
        err = check(0.0015, 100, cir_id, 0) 


