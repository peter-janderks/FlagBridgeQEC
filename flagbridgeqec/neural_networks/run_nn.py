from nn_model_ds import create_model, data_generator_on_the_fly, read_data, data_generator, MyBatchGenerator
import numpy as np
import tensorflow as tf
from flagbridgeqec.sim.sequential.flag_steane import Steane_FT,check_lut, check_hld
from parallel_ds_nn_train import check_hld
import multiprocessing as mp
import random
from tensorflow.python.client import device_lib
from scipy import stats

def neural_network_decoder(trials,qeccode,model_name,cpus):

    mp.freeze_support()
    pool = mp.Pool()

    results = []
    for i in range(0, cpus):
        print(i,'i')
        result = pool.apply_async(task_neural_network, args=(i, qeccode, trials, model_name))
        results.append(result)
    pool.close()
    pool.join()
    ler_result = []
    for result in results:
        ler_result.append(result.get())
    total_err = 0
    no_error=0
    for rst in ler_result:
        total_err += rst[0]
        no_error += rst[1]
    print(no_error,'nooo_error')
    total_err = total_err/(trials*cpus)
    return(total_err)

def task_neural_network(pid, qeccode,trials,model):
    np.random.seed(random.randint(10000, 20000))
    err = check_hld(model,trials,qeccode)
    return err

def write_data_to_file(file,ler_m,conf_int_a):
    of = open(file, 'a')
    of.write('\t')
    of.write(str(ler_m))
    of.write('\t')
    of.write(str(conf_int_a))
    of.write('\n')
    of.close()

def std(mean):
    dev = (1-mean)*(mean**2) + mean*((1-mean)**2)
    return np.sqrt(dev)

print(device_lib.list_local_devices())
trials = 200

model_name = 'test_nn'
cpus = 10


err_lo = 0.0008
err_hi = 0.0012
n_point = 5
errs = np.linspace(err_lo, err_hi, n_point)

confidence = 0.999

file_name = 'results/'+ str(model_name) + '.txt'

for i in range(len(errs)):
    per = errs[i]
    qeccode = Steane_FT(per,per,per,'c3_l2',1,0.1)
    total_errors = neural_network_decoder(trials,qeccode,model_name,cpus)
    
    print(total_errors)

    conf_interval = stats.norm.interval(confidence, loc=total_errors, scale=std(total_errors)/float(np.sqrt(cpus*trials)))
    print(conf_interval,'conf_interval')
    write_data_to_file(file_name,total_errors,conf_interval)
