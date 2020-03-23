import multiprocessing as mp
from collections import Counter
import numpy as np
import random
from scipy import stats
from flagbridgeqec.sim.sequential.flag_steane import Steane_FT,check_lut, check_hld
from flagbridgeqec.utils import error_model_2 as em
from flagbridgeqec.utils import read_data
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

# import matplotlib.pyplot as plt

def task_lut(pid, qeccode,trials):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_lut(trials)
    return err

def task_hld(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_hld(trials,ds)
    return err

def task_lld(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_lld(trials,ds)
    return err

def lut_decoder(trials, qeccode,cpus):
    mp.freeze_support()
    pool = mp.Pool()
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_lut, args=(i,qeccode,trials))
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

def hld_decoder(trials,qeccode,ds,cpus):
    mp.freeze_support()
    pool = mp.Pool()
#    cpus = mp.cpu_count()
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_hld, args=(i, qeccode, trials, ds))
        results.append(result)
    pool.close()
    pool.join()
    ler_result = []
    for result in results:
        ler_result.append(result.get())


    total_err = 0
    for rst in ler_result:
        total_err += rst

    total_err = total_err/(trials*cpus)
    return(total_err)


def lld_decoder(trials, qeccode, ds, cpus):
    mp.freeze_support()
    pool = mp.Pool()
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_lld, args=(i, qeccode, trials, ds))
        results.append(result)
    pool.close()
    pool.join()
    ler_result = []
    for result in results:
        ler_result.append(result.get())

    total_err = 0
    for rst in ler_result:
        total_err += rst

    total_err = total_err/(trials*cpus)
    return(total_err)




def std(mean):
    dev = (1-mean)*(mean**2) + mean*((1-mean)**2)
    return np.sqrt(dev)

def create_plot(per,lers_lut,interval_lut,lers_hld,interval_hld,lers_lld, 
                interval_lld,cir_id,ridle):
    plt.style.use("ggplot")
    plt.errorbar(per, lers_lut,yerr=interval_lut,label='lut')
    plt.errorbar(per, lers_hld,yerr=interval_hld,label='hld')
    plt.errorbar(per, lers_lld,yerr=interval_lld,label='lld')
    plt.xlabel("Logical error rate")
    plt.ylabel("Physical error rate")
    plt.title(str(cir_id) + '$p_I = $ ' + str(ridle))
    plt.grid(True)
    tikzplotlib.save("plots/test.tex")

def write_data_to_file(file,ler_m,conf_int_a):
    of = open(file, 'a')
    of.write('\t')
    of.write(str(ler_m))
    of.write('\t')
    of.write(str(conf_int_a))
    of.write('\n')
    of.close()

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
    cpus = int(argv[13])


    if idl:
        idling = True
    else:
        idling = False

    errs = np.linspace(err_lo, err_hi, n_point)

    # Based on the gate choice, import the function
    file_lut = 'data/lut' + 'cir' + str(cir_id)+ '_' + str(nm) + '.txt'
    file_hld = 'data/hld' + 'cir' + str(cir_id)+ '_' + str(nm) + '.txt'
    file_lld = 'data/hld' + 'cir' + str(cir_id)+ '_' + str(nm) + '.txt'

    lers_lut = []
    intervals_lut = np.zeros((2,len(errs)))

    lers_hld = []
    intervals_hld = np.zeros((2,len(errs)))

    lers_lld = []
    intervals_lld = np.zeros((2,len(errs)))

    hld_dataset_file = 'cir_id_' + str(cir_id) + '/hld_' + str(ridle) + 'dataset.txt'
    lld_dataset_file = 'cir_id_' + str(cir_id) + '/lld_' + str(ridle) + 'dataset.txt'

    hld_ds = read_data(hld_dataset_file,4)
    lld_ds = read_data(lld_dataset_file,14)
    of = open(file_lut, 'a')
    of.write('a new round of simulation, rp2 and rpm are ')
    of.write(str(rp2))
    of.write(' and ')
    of.write(str(rpm))
    of.write('\n')
    of.write(str(idling))
    of.write(str(ridle))
    of.write('\n')
    of.close()
    
#    total_runs = trials*cpus
    
    for i in range(len(errs)):
        per = errs[i]
        qeccode = Steane_FT(per, per/float(rp2), per/float(rpm), cir_id,
                            idling, ridle)
        # run the simulation
        rst = lut_decoder(trials,qeccode,cpus)

        corr = rst['I']
        total = rst['I'] + rst['Y'] + rst['X'] + rst['Z']
        ler_m = 1 - float(corr) / float(total)
        sigma = std(ler_m) / float(np.sqrt(total))
        conf_int_lut = stats.norm.interval(confidence, loc=ler_m, scale=sigma)
        lers_lut.append(ler_m)
        intervals_lut[0][i] = conf_int_lut[0]
        intervals_lut[1][i] = conf_int_lut[1]

        rst_hld = hld_decoder(trials, qeccode, hld_ds,cpus)
        lers_hld.append(rst_hld)
        conf_int_hld = stats.norm.interval(confidence, loc=rst_hld, scale=sigma)
        intervals_hld[0][i] = conf_int_hld[0]
        intervals_hld[1][i] = conf_int_hld[1]

        rst_lld = lld_decoder(trials, qeccode, lld_ds,cpus)
        lers_lld.append(rst_lld)
        conf_int_lld = stats.norm.interval(confidence, loc=rst_lld, scale=sigma\
)
        intervals_lld[0][i] = conf_int_lld[0]
        intervals_lld[1][i] = conf_int_lld[1]
        
        write_data_to_file(file_lut,ler_m,conf_int_lut)
        write_data_to_file(file_hld,rst_hld,conf_int_hld)

    of = open(file_lut, 'a')
    of.write('PER')
    of.write('\t')
    of.write(str(errs))
    of.write('\n')
    of.write('LER')
    of.write('\t')
    of.write(str(lers_lut))
    of.write('\n')
    of.write('interval')
    of.write('\t')
    of.write(str(intervals_hld))
    of.write('\n')  
    of.close()

    create_plot(errs,lers_lut,intervals_lut,lers_hld,intervals_hld,
                lers_lld, intervals_lld, cir_id,ridle)
    
