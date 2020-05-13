import multiprocessing as mp
from collections import Counter
import numpy as np
import random
from scipy import stats
from flagbridgeqec.sim.optimized_circuits.run_circuit import FT_protocol, check

def task(pid, per1, cir_id, trials, idling):
    np.random.seed(random.randint(10000, 20000))
    err = check(per1, trials, cir_id, idling)
    return err

def main(per1, trials, cir_id, idling,cpus):
    mp.freeze_support()
    pool = mp.Pool()
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task, args=(i, per1, cir_id, trials, idling))
        results.append(result)
    pool.close()
    pool.join()
    total_errors = 0

    ler_result = []
    for result in results:
        ler_result.append(result.get())
    ler_resulta = {'I': 0, 'Z': 0, 'X': 0, 'Y': 0}
    for rst in ler_result:
        for log in rst:
            ler_resulta[log] += rst[log]

    return(ler_resulta)


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
    pI = float(argv[7])
    cpus = int(argv[8])

    total_trials = trials * cpus
    errs = np.linspace(err_lo, err_hi, n_point)
    file = 'data/pI_' +str(pI) + '/cir_' + str(cir_id)+ str(total_trials) + '.txt'

    lers = []
    intervals = []

#    of = open(file, 'a')
#    of = open(file, 'w+')
#    of.write('a new round of simulation, err_lo and pI are ')
#    of.write('\n')
#    of.write(str(err_lo))
#    of.write(str(pI))
#    of.write('\n')
#    of.close()

    for per in errs:
        # run the simulation

        rst = main(per, trials, cir_id, pI,cpus)
        print(rst)
        corr = rst['I']
        total = rst['I'] + rst['Y'] + rst['X'] + rst['Z']
        ler_m = 1 - float(corr) / float(total_trials)
        sigma = std(ler_m) / float(np.sqrt(total_trials))
        conf_int_lut = stats.norm.interval(confidence, loc=ler_m, scale=sigma)
        of = open(file, 'a')
        of.write(str(ler_m))
        of.write('\t')
        of.write(str(conf_int_lut))
        of.write('\n')
        of.close()



