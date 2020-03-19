import multiprocessing as mp
from collections import Counter
import numpy as np
import random
from scipy import stats
from flag_steane import check
from flagbridgeqec.utils import error_model_2 as em
# import matplotlib.pyplot as plt

def task(pid, per1, per2, perm, cir_id, trials, idling, ridle):
    np.random.seed(random.randint(10000, 20000))
    err = check(per1, per2, perm, cir_id, trials, idling, ridle)
         
    return err

def main(per1, per2, perm, cir_id, trials, idling, ridle):
    mp.freeze_support()
    pool = mp.Pool()
    cpus = mp.cpu_count()
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task, args=(i, per1, per2, perm, cir_id, trials, idling, ridle))
        results.append(result)
    pool.close()
    pool.join()
    ler_result = []
    for result in results:
        ler_result.append(result.get())
    ler_resulta = {'I': 0, 'Z': 0, 'X': 0, 'Y': 0}

    for rst in ler_result:
        for log in rst:
            ler_resulta[log] += rst[log]

    return ler_resulta


def std(mean):
    dev = (1-mean)*(mean**2) + mean*((1-mean)**2)
    return np.sqrt(dev)


if __name__ == '__main__':
    from sys import argv

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
    cir_id = str(argv[9])
    idl = int(argv[10])
    ridle = float(argv[11])
    nm = str(argv[12])

    if idl:
        idling = True
    else:
        idling = False

    errs = np.linspace(err_lo, err_hi, n_point)

    # Based on the gate choice, import the function
    

    file = '../../runs/' + 'cir' + str(cir_id)+ '_' + str(nm) + '.txt'

    lers = []
    intervals = []

    of = open(file, 'a')
    of.write('a new round of simulation, rp2 and rpm are ')
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
        rst = main(per, per/float(rp2), per/float(rpm), cir_id, trials, idling, ridle)

        corr = rst['I']
        total = rst['I'] + rst['Y'] + rst['X'] + rst['Z']
        ler_m = 1 - float(corr) / float(total)
        sigma = std(ler_m) / float(np.sqrt(total))
        conf_int_a = stats.norm.interval(confidence, loc=ler_m, scale=sigma)
        lers.append(ler_m)
        intervals.append(conf_int_a)
        of = open(file, 'a') 
        of.write(str(per))
        of.write(str(rst))
        of.write('\t')
        of.write(str(ler_m))
        of.write('\t')
        of.write(str(conf_int_a))
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
    of.close()
