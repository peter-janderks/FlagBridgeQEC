src_path =  "../src/"
import sys
sys.path.insert(0, src_path);
import random
import numpy as np
import keras
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam
from keras.objectives import binary_crossentropy
#from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from flagbridgeqec.sim.sequential.flag_steane import Steane_FT,check_lut, check_hld



def e_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def create_model(in_dim, out_dim, number_of_samples=None, decay_rate=None, hidden_sizes=[4], hidden_act='tanh', act='sigmoid', loss='binary_crossentropy',
                 learning_rate=0.002, batchnorm=0, decayrate=False):
    
    with tf.device('/cpu:0'):
        model = Sequential()

        model.add(Dense(int(hidden_sizes[0]), input_dim=in_dim, kernel_initializer='glorot_normal'))
        if batchnorm:
            model.add(BatchNormalization(momentum=batchnorm))
        model.add(Activation(hidden_act))

        for s in hidden_sizes[1:]:
            model.add(Dense(int(s), kernel_initializer='glorot_normal'))
            if batchnorm:
                model.add(BatchNormalization(momentum=batchnorm))
            model.add(Activation(hidden_act))
            print('checking')

        model.add(Dense(out_dim, kernel_initializer='glorot_normal'))
        if batchnorm:
            model.add(BatchNormalization(momentum=batchnorm))
        model.add(Activation(act))
        
    if decay_rate:
        initial_learning_rate = learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=decay_rate,
            staircase=True)
    else:
        lr_schedule=learning_rate


    losses = {'e_binary_crossentropy': e_binary_crossentropy}
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=lr_schedule),#, decay = 1e-2),
                  metrics=['binary_accuracy']
                 )

    return model



def data_generator(qeccode, dataset, batch_size=10, size=20000):

    in_dim = 2*qeccode.num_anc
    out_dimX = qeccode.num_data
    out_dimZ = qeccode.num_data

    c = 0
    while True:
        errors = np.empty((batch_size, out_dimZ+out_dimX), dtype=int) 
        synds = np.empty((batch_size, in_dim), dtype=int) 
        
        for i in range(batch_size):
            # non-deterministic
            # synd_err = qeccode.err_synd()
            # deterministic
            l =np.random.randint(size)
            synds[i,:] = dataset[0][l]
            errors[i,:] = dataset[1][l]
#            syndrome, error = random.choice(list(dataset.items()))

#            for j in range(in_dim):
#                synds[i,j] = int(syndrome[j])
#            for j in range(out_dimZ+out_dimX):
#                errors[i,j] = error[j]

        yield (synds, errors)
        c += 1
        if size and c==size:
            break
            # raise StopIteration
    return
    # return synds, errors

class MyBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, ds, batch_size):
        self.ds = ds
        self.batch_size = batch_size
        self.errors1 = np.empty((self.batch_size, 14), dtype=int)
        self.synds1 = np.empty((self.batch_size, 24), dtype=int)
        idx = 0
        self.synds1 = self.ds[0][self.batch_size*idx:self.batch_size*(idx+1)]
        self.errors1 = self.ds[1][self.batch_size*idx:self.batch_size*(idx+1)]

        self.errors2 = np.empty((self.batch_size, 14), dtype=int)
        self.synds2 = np.empty((self.batch_size, 24), dtype=int)
        idx = 1
        self.synds2 = self.ds[0][self.batch_size*idx:self.batch_size*(idx+1)]
        self.errors2 = self.ds[1][self.batch_size*idx:self.batch_size*(idx+1)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):   
        if idx == 0:
            synds=self.synds1
            errors=self.errors1
        elif idx == 1:
            synds=self.synds2
            errors=self.errors2
#        errors = np.empty((self.batch_size, 14), dtype=int)
#        synds = np.empty((self.batch_size, 24), dtype=int)
#        synds = self.ds[0][self.batch_size*idx:self.batch_size*(idx+1)]
#        errors = self.ds[1][self.batch_size*idx:self.batch_size*(idx+1)]  
#        print(synds,'synds')
#        print(errors,'errors')
#       print(synds,'synds')
#        print(errors,'errors')
#        for j in range(self.batch_size):
#            synds[j,:] = self.ds[0][self.batch_size*idx+j]
#            errors[j,:] = self.ds[1][self.batch_size*idx+j]
        return(synds, errors)

def data_generator_on_the_fly(qeccode, batch_size=10, size=None):

    in_dim = 2*qeccode.num_anc
    out_dimX = qeccode.num_data
    out_dimZ = qeccode.num_data

    c = 0
    while True:
        errors = np.empty((batch_size, out_dimZ+out_dimX), dtype=int) 
        synds = np.empty((batch_size, in_dim), dtype=int) 
        
        for i in range(batch_size):
            # non-deterministic
            # synd_err = qeccode.err_synd()
            # deterministic
            synd_err = qeccode.err_synd2()
            synds[i,:] = synd_err[0]
            errors[i,:] = synd_err[1]

        # print(synds, errors)
        yield (synds, errors)
        c += 1
        if size and c==size:
            break
            # raise StopIteration
    # return synds, errors



def read_data(filename, inputs):    
    """
    Retrieve the .txt file that contains the dataset as obtained through sampling
    in the following format
    -------inputs---------    outputs    freq
    0 1 0 0 1 0 .... 0 0 0  523 ... 18   541
    """
    with open(filename, 'r') as f:
        m = 0
        for i, l in enumerate(f):
            pass
        no_samples = i+1

        syndromes = np.zeros((no_samples,16))
        errors = np.zeros((no_samples,4))
        f.seek(0) #move to start of file
        for line in f:  
            m += 1
            data_smpl = line.split() 
            data_in = list(data_smpl[0])
#            data_in = [int(d) for d in data_smpl[:-(inputs+1)]]
            data_dumb = np.array([float(d) for d in data_smpl[-(inputs+1):-1]])
            syndromes[m-1] = data_in

            errors[m-1] = data_dumb
            if m % 1000 == 0:
                print(m, '\r', end='')

#            if m == no_samples:
#                break
        f.close()
        
    return([syndromes,errors])

if __name__ == '__main__':
    # print('running')
    
    from sys import argv
    err_lo = 0.1
    err_hi = 0.2
    n_point = 2
    trials = 3
    confidence =0.5
    cd = 'Steane'
    rp2 = 10
    rpm = 15
    cir_id = 2
    idl = False
    ridle = False
    # input parameters                                                                  
#    err_lo = float(argv[1])
#    err_hi = float(argv[2])
#    n_point = int(argv[3])
 #   trials = int(argv[4])
 #   confidence = float(argv[5])
    # cd is the qec code name e.g. steane                                               
 #   cd = str(argv[6])
 #   rp2 = float(argv[7])
 #   rpm = float(argv[8])
 #   cir_id = int(argv[9])
 #   idl = int(argv[10])
 #   ridle = float(argv[11])
    # nm is the model name                                                              
  #  nm = str(argv[12])
    per = err_lo
    idling = False
    qeccode = Steane_FT(per, per/float(rp2), per/float(rpm), cir_id, idling, ridle)
    ff = data_generator(qeccode,batch_size=1,size=1)

    for i in ff:
        print(i)
    # print('finish')
