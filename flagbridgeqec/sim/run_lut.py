import multiprocessing as mp
from collections import Counter
import numpy as np
import random
from scipy import stats
from run_circuit import check
# import matplotlib.pyplot as plt

def task(pid, per1, cir_id, trials, idling):
    np.random.seed(random.randint(10000, 20000))
    err = check(per1, trials, cir_id, idling)
    print(err,'err')
    return err

def main(per1, trials, cir_id, idling):
    mp.freeze_support()
    pool = mp.Pool()
    cpus = 10
    results = []
    for i in range(0, cpus):
        print(trials)
        result = pool.apply_async(task, args=(i, per1, cir_id, trials, idling))
        results.append(result)
    pool.close()
    pool.join()
    total_errors = 0
    for result in results:
        total_errors += result.get()
    total_trials = cpus*trials
    return total_errors, total_trials


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
    cir_id = str(argv[6])
    pI = int(argv[7])
    nm = str(argv[8])

    errs = np.linspace(err_lo, err_hi, n_point)
    file = './runs/flag_cir_' + str(cir_id)+ '_' + str(nm) + '.txt'

    lers = []
    intervals = []

#    of = open(file, 'a')
    of = open(file, 'w+')
    of.write('a new round of simulation, err_lo and pI are ')
    of.write('\n')
    of.write(str(err_lo))
    of.write(str(pI))
    of.write('\n')
    of.close()

    for per in errs:
        # run the simulation
        total_errors,total_trials = main(per, trials, cir_id, pI)
        print(total_errors,'total_errors')
        print(total_trials,'total_trials')
        ler_m = total_errors/total_trials
        
        sigma = std(ler_m) / float(np.sqrt(total_trials))
        conf_int_a = stats.norm.interval(confidence, loc=ler_m, scale=sigma)
        lers.append(ler_m)
        intervals.append(conf_int_a)
        of = open(file, 'a')
        of.write(str(total_trials))
        of.write('\n')
        of.write(str(per))
#        of.write(str(rst))
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
