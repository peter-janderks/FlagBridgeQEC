import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import tikzplotlib


def add_to_array(filename):
    with open(filename, 'r') as f:
        ler = np.zeros((11))
        line_count = 0
        for line in f:
            data_smpl = line.split()
            ler[line_count] = float(data_smpl[0])
            line_count +=1
    return(ler)

def std(mean):
    dev = (1-mean)*(mean**2) + mean*((1-mean)**2)
    return np.sqrt(dev)


def create_plot(ler, ds_size_one, ridle):
    plt.style.use("ggplot")
    fig, host = plt.subplots()
    print(ler,'ler')
    for i in range(len(ler)):
        print(i,'i')
        print(ler[i],'ler[i]')
        interval_one =  (stats.norm.interval(0.999, loc=ler[i],
                                             scale=std(ler[i])/float(np.sqrt(1000000))))
        interval_one  = interval_one[1] - ler[i]
        
        host.errorbar(ds_size_one,ler[i],yerr=interval_one,capsize=10)

    tikzplotlib.save('plots/pI_' + str(ridle) + '_tokyo.tex')

ridle = 1.0

if ridle == 1.0 or ridle == 0.1:
    per = 0.001
else:
    per = 0.005

fn = './data/pI_' + str(ridle) + '/cir_c1_l21000000.txt'
ds_array_four = add_to_array(fn)

fn = './data/pI_' + str(ridle) + '/cir_c2_l21000000.txt'
ds_array_five = add_to_array(fn)

fn = './data/pI_' + str(ridle) + '/cir_c3_l21000000.txt'
ds_array_six = add_to_array(fn)

fn = './data/pI_' + str(ridle) + '/cir_4a1000000.txt'
ds_array_one = add_to_array(fn)

#fn = './data/pI_' + str(ridle) + '/cir_5a1000000.txt'
#ds_array_two = add_to_array(fn)

fn = './data/pI_' + str(ridle) + '/cir_6a1000000.txt'
ds_array_three = add_to_array(fn)


if ridle == 1.0 or ridle == 0.1: 
    per = np.linspace(0.0005,0.0015,11)
else:
    per = np.linspace(0.001,0.002,11)

create_plot([ds_array_four, ds_array_five, ds_array_six, ds_array_one,ds_array_three],per,ridle)
