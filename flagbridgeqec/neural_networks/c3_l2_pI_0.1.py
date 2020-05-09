#from tensorflow.python.client import device_lib
from nn_model_ds import create_model, data_generator_on_the_fly, read_data, data_generator, MyBatchGenerator
import numpy as np
import tensorflow as tf
from flagbridgeqec.sim.sequential.flag_steane import Steane_FT,check_lut, check_hld
from parallel_ds_nn_train import check_hld
import multiprocessing as mp
import random

class neural_network(object):
    def __init__(self,argv):
        self.argv = argv
        self.per = float(argv[1])
        self.cir_id = str(argv[2])
        self.idling = int(argv[3])
        self.ridle = float(argv[4])

        qeccode = Steane_FT(self.per,self.per,self.per,self.cir_id,self.idling,self.ridle)
        
        in_dim = 2*qeccode.num_anc
        out_dim = 4
        layers = list(argv[5])
        hidden_activation_function = str(argv[6])
        output_activation_function = str(argv[7])
        loss_function = str(argv[8])
        model_name = 'test_nn'
        lr = 0.002
#        batchsize= 1000
        n_epochs = 1000
        cpus = 10
        trials = 10
        model = create_model(in_dim=in_dim, out_dim=out_dim,
                             hidden_sizes=layers,
                             hidden_act=hidden_activation_function,
                             act=output_activation_function,
                             loss=loss_function,
                             learning_rate=lr)
        dataset_file = '../datasets/cir_id_' + str(cir_id) + '/pI_' + str(ridle) + '/hld_100000000.0010.1dataset.txt'
        ds = read_data(dataset_file, 4)

        hist = model.fit(x=ds[0],
                 y=ds[1],
                 batch_size = 100,
                 epochs = n_epochs,
                 verbose=2,
                 use_multiprocessing=True,
                 workers=cpus,
                 max_queue_size=cpus)
        layers_string ='_'.join((str(n)) for n in layers)
        model.save('nn_model/'+ str(layers_string) +'.model')

def neural_network_decoder(trials,qeccode,model_name,cpus):
#    err = check_hld(model, 100, qeccode)
    pid = 10
#    errors = task_neural_network(pid, qeccode,trials,model_name)

    mp.freeze_support()
    pool = mp.Pool()
    cpus = mp.cpu_count()
    print(cpus,'cpus')
    results = []
    for i in range(0, 1):
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

if __name__ == '__main__':
    from sys import argv
    if len(argv) >= 2:
        network = neural_network(argv)
    else:
        per = 0.001
        cir_id = 'c3_l2'
        idling = 1
        ridle = 0.1
        layers = [16,12]
        hidden_activation_function = 'tanh'
        output_activation_function = 'softmax'
        loss_function = 'binary_crossentropy'

        input_array = [0,per,cir_id,idling,ridle,layers,hidden_activation_function,output_activation_function, loss_function]
        network = neural_network(input_array)
        

