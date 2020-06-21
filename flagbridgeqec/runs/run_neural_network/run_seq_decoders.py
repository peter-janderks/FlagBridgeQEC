import multiprocessing as mp
from collections import Counter
import numpy as np
import random
from scipy import stats
from flagbridgeqec.sim.sequential.flag_steane import Steane_FT,check_lut, check_hld
from flagbridgeqec.utils import error_model_2 as em
from flagbridgeqec.utils import read_data, read_data_perfect_lld
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

# import matplotlib.pyplot as plt
class Run_Decoder(object):
    def __init__(self, argv):
        self.argv = argv
        self.err_lo = float(argv[1])
        self.err_hi = float(argv[2])
        self.n_point = int(argv[3])
        self.trials = int(argv[4])
        self.confidence = float(argv[5])
        self.rp2 = float(argv[7])
        self.rpm = float(argv[8])
        self.cir_id = str(argv[9])
        self.idl = int(argv[10])
        self.ridle = float(argv[11])
        self.nm = str(argv[12])
        self.cpus = int(argv[13])
        self.total_trials = self.cpus*self.trials
        self.number_of_decoders = len(argv)-14
        self.datasets = []
        self.result_files=[]
        self.ler = np.empty(shape=(self.number_of_decoders,self.n_point))
        self.intervals = np.zeros((self.number_of_decoders,2,self.n_point))
        self.errs = np.linspace(self.err_lo, self.err_hi, self.n_point)
        if self.idl:
            self.idling = True
        else:
            self.idling = False
            
        for i in range(len(argv)-14):
            self.datasets.append(self.prep_dataset(argv[14+i]))
            self.result_files.append('new_data/'+'cir' + str(self.cir_id) +'/pI_' + str(self.ridle)+ '/' + str(argv[14+i]) + str(self.total_trials) + str(self.nm) + '.txt')

        for i in range(self.n_point):

            per = self.errs[i]
            print(per, 'per')
            print(self.idling,'self.idling')
            print(self.ridle,'ridle')

            qeccode = Steane_FT(per, per/float(self.rp2), per/float(self.rpm), self.cir_id,
                                self.idling, self.ridle)

            for j in range(len(argv)-14):
                print(j,'j')
                self.ler[j,i] = self.decode(qeccode,self.datasets[j],argv[14+j])
                conf_interval = stats.norm.interval(self.confidence, loc=self.ler[j,i], 
                                                   scale=std(self.ler[j,i])/float(np.sqrt(self.cpus*self.trials)))
                write_data_to_file(self.result_files[j],self.ler[j,i],conf_interval)
        
        print(self.ler,'ler')
        self.create_plot(self.ler[0])

    def prep_dataset(self,name):
        dataset_file = 'cir_id_' + str(self.cir_id) + '/pI_' + str(self.ridle) + '/' + str(name) + 'dataset.txt'
        if name == 'lld_':
            ds = read_data(dataset_file,14)
        elif name[0:11] == 'perfect_lld':
            ds = read_data_perfect_lld(dataset_file,14)
        else:
            ds = read_data(dataset_file,4)
        return(ds)
    
    def decode(self,qeccode,ds,name):
        print(name,'name')
        if name == 'lld_':
            task = task_lld

        elif name[0:11] == 'perfect_lld' or name[0:11] == 'lld_experim':
            print('yes')
            task = task_perfect_lld
        elif name[0:5] == 'hld_3':
            task = task_hld_3
        elif name[0:5] == 'hld_4':
            task = task_hld_4
        elif name[0:5] == 'hld_5':
            task = task_hld_5
        elif name[0:4] == 'hld_':
            task = task_hld
        
        mp.freeze_support()
        pool = mp.Pool()

        results = []
        for i in range(0, self.cpus):
            result = pool.apply_async(task, args=(i,qeccode, self.trials, ds))
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
        total_err = total_err/(self.trials*self.cpus)
        print(total_err)
        return(total_err)
    
    def create_plot(self,ler):
        plt.style.use("ggplot")
        fig, host = plt.subplots()
        host.plot(self.errs,ler)
        print('DONE')
        tikzplotlib.save("plots/chicken.tex")

def task_lut(pid, qeccode,trials):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_lut(trials)
    return err

def task_hld(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_hld(trials,ds)
    return err

def task_hld_2(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_hld_2(trials,ds)
    return err

def task_hld_3(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_hld_3(trials,ds)
    return err

def task_hld_4(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_hld_4(trials,ds)
    return err

def task_hld_5(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_hld_5(trials,ds)
    return err

def task_lld(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_lld_2(trials,ds)
    return err

def task_perfect_lld(pid, qeccode,trials,ds):
    np.random.seed(random.randint(10000, 20000))
    err = qeccode.run_perfect_lld(trials,ds)
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
    no_error=0
    for rst in ler_result:
        total_err += rst[0]
        no_error += rst[1]
    print(no_error,'nooo_error')
    total_err = total_err/(trials*cpus)
    return(total_err)

def hld_2_decoder(trials,qeccode,ds,cpus):
    mp.freeze_support()
    pool = mp.Pool()

    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_hld_2, args=(i, qeccode, trials, ds))
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
        no_error
    print(no_error,'nooo_error')
    total_err = total_err/(trials*cpus)
    return(total_err)

def hld_3_decoder(trials,qeccode,ds,cpus):
    mp.freeze_support()
    pool = mp.Pool()

    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_hld_3, args=(i, qeccode, trials, ds))
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


def hld_4_decoder(trials,qeccode,ds,cpus):
    mp.freeze_support()
    pool = mp.Pool()

    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_hld_4, args=(i, qeccode, trials, ds))
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

def hld_5_decoder(trials,qeccode,ds,cpus):
    mp.freeze_support()
    pool = mp.Pool()

    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task_hld_5, args=(i, qeccode, trials, ds))
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
    no_error = 0
    for rst in ler_result:
        total_err += rst[0]
        no_error += rst[1]
    print(no_error,'nooo_error')
    total_err = total_err/(trials*cpus)
    return(total_err)

def std(mean):
    dev = (1-mean)*(mean**2) + mean*((1-mean)**2)
    return np.sqrt(dev)

def create_plot(per,lers_lut,interval_lut,lers_hld,interval_hld,lers_hld_4,interval_hld_4,lers_hld_5,interval_hld_5,lers_lld, 
                interval_lld,cir_id,ridle,name,trials):
    plt.style.use("ggplot")
    fig, host = plt.subplots()

    interval =  (stats.norm.interval(0.999, loc=lers_lut, scale=std(lers_lut)/float(np.sqrt(trials))))
    interval_lut  = interval[1] - lers_lut

    interval = (stats.norm.interval(0.999, loc=lers_lld, scale=std(lers_lld)/float(np.sqrt(trials))))
    interval_lld = interval[1] - lers_lld

    interval = (stats.norm.interval(0.999, loc=lers_hld, scale=std(lers_hld)/float(np.sqrt(trials))))
    interval_hld = interval[1] - lers_hld

    interval = (stats.norm.interval(0.999, loc=lers_hld_5, scale=std(lers_hld_5)/float(np.sqrt(trials))))
    interval_hld_5 = interval[1] - lers_hld_5

    interval = (stats.norm.interval(0.999, loc=lers_hld_4, scale=std(lers_hld_4)/float(np.sqrt(trials))))
    interval_hld_4 = interval[1] - lers_hld_4

    lut = host.errorbar(per, lers_lut,yerr=interval_lut,label='lut',capsize=10)
    hld = host.errorbar(per, lers_hld,yerr=interval_hld,label='hld',capsize=10)
    hld_5 = host.errorbar(per, lers_hld_5,yerr=interval_hld_5,label='hld_5',capsize=10)
    hld_4 = host.errorbar(per, lers_hld_4,yerr=interval_hld_4,label='hld_4',capsize=10)
    lld = host.errorbar(per, lers_lld,yerr=interval_lld,label='lld',capsize=10)


    host.set_xlabel("Logical error rate")
    host.set_ylabel("Physical error rate")
    host.set_title(str(cir_id) + '$p_I = $ ' + str(ridle))
    host.grid(True)
#    host.legend([lut,hld,lld],loc='upper left')
    tikzplotlib.save("plots/" + name + ".tex")

def write_data_to_file(file,ler_m,conf_int_a):
    of = open(file, 'a')
    of.write('\t')
    of.write(str(ler_m))
    of.write('\t')
    of.write(str(conf_int_a))
    of.write('\n')
    of.close()

def run_decoder(name):
    data_file= 'new_data/' + str(name) + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    lers = np.empty(shape=(n_point))
    intervals = np.zeros((2,len(errs)))
    dataset_file = 'cir_id_' + str(cir_id) + '/' + str(name) + str(ridle) + 'dataset.txt'
    if name == 'lld':
        ds = read_data(dataset_file,14)
    else:
        ds = read_data(dataset_file,4)
    
if __name__ == '__main__':
    from sys import argv

    x = Run_Decoder(argv)

    print(stop)
    # input parameters
#    err_lo = float(argv[1])
#    err_hi = float(argv[2])
#    n_point = int(argv[3])
#    trials = int(argv[4])
#    confidence = float(argv[5])
    # cd is the qec code name e.g. steane
#    cd = str(argv[6])
#    rp2 = float(argv[7])
#    rpm = float(argv[8])
    cir_id = str(argv[9])
    idl = int(argv[10])
    ridle = float(argv[11])
    nm = str(argv[12])
    cpus = int(argv[13])


    if idl:
        idling = True
    else:
        Idlings = False

    errs = np.linspace(err_lo, err_hi, n_point)

    # Based on the gate choice, import the function
    file_lut = 'data/lut' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    file_hld = 'data/hld' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    file_hld_2 = 'data/hld_2' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    file_hld_3 = 'data/hld_3' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    file_hld_4 = 'data/hld_4' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    file_hld_5 = 'data/hld_5' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'
    file_lld = 'data/lld' + 'cir' + str(cir_id)+ '_' + str(ridle) + str(nm) + '.txt'


    lers_lut = np.empty(shape=(n_point))
    intervals_lut = np.zeros((2,len(errs)))

    lers_hld = np.empty(shape=(n_point))
    intervals_hld = np.zeros((2,len(errs)))

    lers_hld_2 = np.empty(shape=(n_point))
    intervals_hld_2 = np.zeros((2,len(errs)))

    lers_hld_3 = np.empty(shape=(n_point))
    intervals_hld_3 = np.zeros((2,len(errs)))

    lers_hld_4 = np.empty(shape=(n_point))
    intervals_hld_4 = np.zeros((2,len(errs)))

    lers_hld_5 = np.empty(shape=(n_point))
    intervals_hld_5 = np.zeros((2,len(errs)))

    lers_lld = np.empty(shape=(n_point))
    intervals_lld = np.zeros((2,len(errs)))

    hld_dataset_file = 'cir_id_' + str(cir_id) + '/hld_' + str(ridle) + 'dataset.txt'
    hld_2_dataset_file = 'cir_id_' + str(cir_id) + '/hld_2' + str(ridle) + 'dataset.txt'
    hld_3_dataset_file = 'cir_id_' + str(cir_id) + '/hld_3' + str(ridle) + 'dataset.txt'
    hld_4_dataset_file = 'cir_id_' + str(cir_id) + '/hld_4' + str(ridle) + 'dataset.txt'
    hld_5_dataset_file = 'cir_id_' + str(cir_id) + '/hld_5' + str(ridle) + 'dataset.txt'

    lld_dataset_file = 'cir_id_' + str(cir_id) + '/lld_' + str(ridle) + 'dataset.txt'

#    hld_2_ds = read_data(hld_2_dataset_file,4)

    hld_ds = read_data(hld_dataset_file,4)
    hld_3_ds = read_data(hld_3_dataset_file,4)    
    hld_4_ds = read_data(hld_4_dataset_file,4)
    hld_5_ds = read_data(hld_5_dataset_file,4)

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
        lers_lut[i] = ler_m
        intervals_lut[0][i] = conf_int_lut[0]
        intervals_lut[1][i] = conf_int_lut[1]

        rst_hld = hld_decoder(trials, qeccode, hld_ds,cpus)
        lers_hld[i] = rst_hld
        conf_int_hld = stats.norm.interval(confidence, loc=rst_hld, scale=sigma)
        intervals_hld[0][i] = conf_int_hld[0]
        intervals_hld[1][i] = conf_int_hld[1]

        rst_hld_3 = hld_3_decoder(trials, qeccode, hld_3_ds,cpus)                                 
        lers_hld_3[i] = rst_hld_3
        conf_int_hld_3 = stats.norm.interval(confidence, loc=rst_hld_3, scale=sigma)            
        intervals_hld_3[0][i] = conf_int_hld_3[0]
        intervals_hld_3[1][i] = conf_int_hld_3[1] 

        rst_hld_4 = hld_4_decoder(trials, qeccode, hld_4_ds,cpus)
        lers_hld_4[i] = rst_hld_4
        conf_int_hld_4 = stats.norm.interval(confidence, loc=rst_hld_4, scale=sigma)
        intervals_hld_4[0][i] = conf_int_hld_4[0]
        intervals_hld_4[1][i] = conf_int_hld_4[1]

        rst_hld_5 = hld_5_decoder(trials, qeccode, hld_5_ds,cpus)
        lers_hld_5[i] = rst_hld_5
        conf_int_hld_5 = stats.norm.interval(confidence, loc=rst_hld_5, scale=sigma)
        intervals_hld_5[0][i] = conf_int_hld_5[0]
        intervals_hld_5[1][i] = conf_int_hld_5[1]

#        rst_hld_2 = hld_2_decoder(trials, qeccode, hld_2_ds,cpus)
#        lers_hld_2[i] = rst_hld_2
#        conf_int_hld_2 = stats.norm.interval(confidence, loc=rst_hld_2, scale=sigma)
#        intervals_hld_2[0][i] = conf_int_hld_2[0]
#        intervals_hld_2[1][i] = conf_int_hld_2[1]


        rst_lld = lld_decoder(trials, qeccode, lld_ds,cpus)
        lers_lld[i] = rst_lld
        conf_int_lld = stats.norm.interval(confidence, loc=rst_lld, scale=sigma)
        intervals_lld[0][i] = conf_int_lld[0]
        intervals_lld[1][i] = conf_int_lld[1]
        
        write_data_to_file(file_lut,ler_m,conf_int_lut)
        write_data_to_file(file_hld,rst_hld,conf_int_hld)
        write_data_to_file(file_hld_3,rst_hld_3,conf_int_hld_3)
        write_data_to_file(file_hld_4,rst_hld_4,conf_int_hld_4)
        write_data_to_file(file_hld_5,rst_hld_5,conf_int_hld_5)
        write_data_to_file(file_lld,rst_lld,conf_int_lld)

  #      write_data_to_file(file_hld_2,rst_hld_2,conf_int_hld_2)


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

    Plot_name = 'cir_id' + str(cir_id) + 'pI' + str(ridle) + str(nm)
    create_plot(errs,lers_lut,intervals_lut,lers_hld,intervals_hld,lers_hld_4,intervals_hld_4,lers_hld_5,intervals_hld_5, 
                lers_lld, intervals_lld, cir_id,ridle,plot_name,total)
    
