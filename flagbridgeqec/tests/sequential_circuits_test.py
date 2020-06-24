import pytest
from flagbridgeqec.sim.sequential.flag_steane import check_lut
from flagbridgeqec.sim.sequential.check_ft1 import check

circuits = ['c1_l1','c2_l1', 'c1_l2', 'c2_l2', 'c3_l2']

def test_if_circuits_are_ft():
    for i in circuits:
        ft_value = check(i, steane=True)
        assert ft_value == 1

def test_lut_decoder_sequential_circuit():
    for i in circuits:
        err = check_lut(0.0115, 0.0015, 0.0015, i, 10, idling=False, ridle=0)


#    assert 1. == pytest.approx(
#        eng.backend.get_probability([0, 0, 1, 0, 0], qunum_a))

