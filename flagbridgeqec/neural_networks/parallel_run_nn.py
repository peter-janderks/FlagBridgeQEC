
import multiprocessing as mp
from collections import Counter
import numpy as np
import random
from scipy import stats
from parallel_ds_nn_train import check
src_path =  "../src/"
import sys
sys.path.insert(0, src_path);
from flag_steane import Steane_FT


# import matplotlib.pyplot as plt

def task(outputname, trials, qeccode):
    np.random.seed(random.randint(10000, 20000))
    err_num = check(outputname, trials, qeccode)         
    return err_num

def main(outputname, trials, qeccode):
    mp.freeze_support()
    pool = mp.Pool()
    cpus=10
#    cpus = mp.cpu_count()
    used_cpus = cpus
    results = []
    print('testing')
    
    for i in range(0, used_cpus):
        result = pool.apply_async(task, args=(outputname, trials, qeccode))
        results.append(result)
    pool.close()
    pool.join()

    total_err = 0
    for result in results:
        total_err += result.get()

    return total_err, used_cpus*trials


def std(mean):
    dev = (1-mean)*(mean**2) + mean*((1-mean)**2)
    return np.sqrt(dev)


if __name__ == '__main__':
    from sys import argv
    import tensorflow as tf 
    print(tf.__version__)
    # input parameters
    err_lo = float(argv[1])
    err_hi = float(argv[2])
    n_point = int(argv[3])
    trials = int(argv[4])
    confidence = float(argv[5])
    # cd is the qec code name e.g. steane
    cd = str(argv[6])
    rp2 = float(argv[7])
    rpm = float(argv[8])
    cir_id = int(argv[9])
    idl = int(argv[10])
    ridle = float(argv[11])
    # nm is the model name
    nm = str(argv[12])

    if idl:
        idling = True
    else:
        idling = False

    errs = np.linspace(err_lo, err_hi, n_point)

    # Based on the gate choice, import the function
    

    file = str(nm) + 'cir' + str(cir_id)+ '_' + '.txt'

    lers = []
    intervals = []

    of = open(file, 'w+')
    of.write('a new round of simulation, using nnmodle ')
    of.write(str(nm))
    of.write(' \n rp2 and rpm are ')
    of.write(str(rp2))
    of.write(' and ')
    of.write(str(rpm))
    of.write('\n')
    of.write(str(idling))
    of.write(str(ridle))
    of.write('\n')
    of.close()

    for per in errs:
        # run the simulation
        qeccode = Steane_FT(per, per/float(rp2), per/float(rpm), cir_id, idling, ridle)
#        model = tf.keras.models.load_model(nm+'.model') 
#        model = tf.keras.models.load_model('../cir_id2/small_dataset/' + nm+'.model')
#        synds = np.zeros((1,24))
#        pred = model.predict(synds).ravel()
#        print(pred,'pred')
        #line70
#        synds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
#        synds =np.reshape(synds,(1, 24))
#        pred = model.predict(synds).ravel()
#        print('line2')
#        print(pred,'pred')
        
        #line42
#        synds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0])
#        synds =np.reshape(synds,(1, 24))
#        pred = model.predict(synds).ravel()
#        print(pred,'pred')
        #line42                                                              
#        synds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
#        synds =np.reshape(synds,(1, 24))
#        pred = nnmodel.predict(synds).ravel()
#        print(pred,'pred')
#        print(pred[3], 'line 42')

 #       synds = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, #0, 1, 1, 0, 0, 0])
 #       synds =np.reshape(synds,(1, 24))
 #       pred = nnmodel.predict(synds).ravel()
 #       print(pred,'pred')
 #       print(pred[3], 'line 498 should be 0.85')


        total_err, total_trials = main(outputname=nm, trials=trials, qeccode=qeccode)
                
        print(total_err)
        print(total_trials,'total_trials')
        ler_m = float(total_err)/float(total_trials)
        sigma = std(ler_m) / float(np.sqrt(total_trials))
        conf_int_a = stats.norm.interval(confidence, loc=ler_m, scale=sigma)
        lers.append(ler_m)
        intervals.append(conf_int_a)
        of = open(file, 'a') 
        of.write(str(per))
        of.write(': ')
        of.write(str(ler_m))
        of.write('\t')
        of.write(str(conf_int_a))
        of.write('\n')
        of.write('total runs')
        of.write(str(total_trials))
        of.write('\n')
        of.close() 

    of = open(file, 'a')
    of.write('PER')
    of.write('\t')
    of.write(str(errs))
    of.write('\n')
    of.write('LER')
    of.write('\t')
    of.write(str(lers))
    of.write('\n')
    of.write('interval')
    of.write('\t')
    of.write(str(intervals))
    of.write('\n')  
    of.write(str(argv))
    of.close()
