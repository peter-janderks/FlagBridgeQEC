#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import sparse_pauli as sp
from flagbridgeqec.circuits import esm2, esm3, esm4, esm5, esm7
from flagbridgeqec.sim.sequential.check_ft1 import Check_FT
import flagbridgeqec.sim.sequential.check_ft1 as cf
from flagbridgeqec.sim.sequential.flag_steane import circ2err,fowler_model, singlelogical_error_index

from flagbridgeqec.utils import error_model_2 as em
import circuit_metric as cm
from operator import mul
from functools import reduce
import numpy as np
from tensorflow.python.client import device_lib
from nn_model_ds import create_model, data_generator_on_the_fly, read_data, data_generator, MyBatchGenerator

import numpy as np
src_path =  "../src/"
#import sys
#sys.path.insert(0, src_path);
#from flag_steane import Steane_FT
#import tensorflow as tf

def check_hld(outputname, trials, qeccode):
    import tensorflow as tf

    with tf.device('/CPU:0'):
        model = tf.keras.models.load_model('nn_model/'+outputname+'.model')
        no_error = 0
        total_errors = 0
        for trial in range(trials):
            for key in qeccode.datas:
                qeccode.fnl_errsx[key] = 0
                qeccode.fnl_errsz[key] = 0
            synd_list = np.zeros((qeccode.num_anc*2))
            synd_zeros = np.zeros((qeccode.num_anc*2))
            corr = sp.Pauli()
        
            err,synd1 = qeccode.run_first_round()
            if len(synd1):
                err, synd2,corr = qeccode.run_second_round(err,synd1)
            else:
                synd2 = dict()
                no_error += 1

            err_fnl = err
            err *= corr

            synd_fnl = set()
            for i in range(len(qeccode.esm_circuits)):
                subcir = qeccode.esm_circuits[i]
                synd_err = circ2err(subcir, fowler_model(subcir, 0, 0, 0), err, qeccode.ancillas)
                synd_fnl |= synd_err[0]
                err = synd_err[1]
            err *= qeccode.lut_synd[tuple(sorted(synd_fnl & set(qeccode.q_syndx)))]
            err *= qeccode.lut_synd[tuple(sorted(synd_fnl & set(qeccode.q_syndz)))]
        
            err_tp = singlelogical_error_index(err, qeccode.init_logs)


            for key in err_fnl.x_set:
                qeccode.fnl_errsx[key] = 1
            for key in err_fnl.z_set:
                qeccode.fnl_errsz[key] = 1

            for i in range(qeccode.num_anc):
                if qeccode.ancillas[i] in synd1:
                    if qeccode.ancillas[i] in set(qeccode.q_flag):
                        synd_list[i] = 1

            for j in range(qeccode.num_anc):
                if qeccode.ancillas[j] in synd2:
                    synd_list[j+qeccode.num_anc] = 1

            if (synd_list != synd_zeros).any():
                synd_list =np.reshape(synd_list,(1, 16))
                pred = model.predict(synd_list)
                if err_tp != np.argmax(pred):
                    total_errors +=1
    print(total_errors)
    return(total_errors,no_error)


def check(outputname, trials, qeccode):
    # evaluation function
    import tensorflow as tf
    print(device_lib.list_local_devices())
    print('here')
    with tf.device('/CPU:0'):
#        model = tf.keras.models.load_model('../cir_id2/small_dataset/' + outputname+'.model')
        model = tf.keras.models.load_model(outputname+'.model')

        print('yes, evaluation is running')
        c = 0
        synds_errgen = data_generator_on_the_fly(qeccode, batch_size=1, size=trials)
    
        for i, (synds, errs) in enumerate(synds_errgen):
            errs = errs.ravel()
            pred = model.predict(synds).ravel()

            sample = np.round(pred)
            left_errs = (sample+errs)%2
            syndx_fnl = qeccode.Hx.dot(left_errs)%2
            syndz_fnl = qeccode.Hz.dot(left_errs)%2
            errors = qeccode.E.dot(left_errs)%2 # what is this?
            if np.any(syndx_fnl):
                if errors[0] == 0:
                    c += 1
                elif np.any(syndz_fnl):
                    if errors[1] == 0: 
                        c += 1
                else:
                    if errors[1]:
                        c += 1

            elif np.any(syndz_fnl):
                    
                if errors[1] == 0: 
                    c += 1
                elif errors[0]:
                    c += 1
            else:
                if np.any(errors):
                    c += 1
                
        # print(c)
        
    return c
    # with open('nn_model/'+ outputname+'.eval', 'a') as f:
    #     f.write('\n')
    #     f.write(str(args))
    #     f.write('\n LER is ')
    #     f.write(str(((c/trials),)))
    #     print(c/trials)

#class Histories(tf.keras.callbacks.Callback):

#    def on_train_begin(self,logs={}):
#        print('test')
#        self.losses = []
#        self.accuracies = []
#        self.learning_rate = []
#    def on_batch_end(self, batch, logs={}):
#        print(model.optimizer.get_config())
#        print(model.optimizer.get_gradients())
#        print(model.optimizer.get_updates())
#        self.losses.append(logs.get('loss'))
#        self.accuracies.append(logs.get('binary_accuracy'))
#        self.learning_rate.append(model.optimizer.get_config())

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Train a neural network to decode a code.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='''\
    ''')
    parser.add_argument('cir_id', type=int, default=2, help='the circut id of steane code error correction(default: %(default)s)')
    parser.add_argument('out', type=str,
                        help='the name of the output file (used as a prefix for the log file as well)')
    parser.add_argument('--trainset', type=str,
                        help='the name of the training set file (generated by `generate_training_data.py`); if not specified --onthefly is assumed')
    parser.add_argument('--onthefly', type=int, nargs=2, default=[100000, 10000],
                        help='generate the training set on the fly, specify training and validation size (default: %(default)s)')
    parser.add_argument('--prob', type=float, nargs=3, default=[0.01, 0.01, 0.01],
                        help='the probability of an error on the physical qubit when generating training data (considered only if --onthefly is present) (default: %(default)s)')
    parser.add_argument('--eval', action='store_true',
                        help='if present, calculate the fraction of successful corrections based on sampling the NN using the validation set')
    parser.add_argument('--batch', type=int, default=512,
                        help='the batch size (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=0,
                        help='the number of epochs (default: %(default)s)')
    parser.add_argument('--learningrate', type=float, default=0.002,
                        help='the learning rate (default: %(default)s)')
    parser.add_argument('--hact', type=str, default='tanh',
                        help='the activation for hidden layers (default: %(default)s)')
    parser.add_argument('--act', type=str, default='sigmoid',
                        help='the activation for the output layer (default: %(default)s)')
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        help='the loss to be optimized (default: %(default)s)')
    parser.add_argument('--layers', type=float, default=[4, 4, 4], nargs='+',
                        help='the list of sizes of the hidden layers (as a factor of the output layer) (default: %(default)s)')
    parser.add_argument('--decayrate', type=float, default=False,
                        help='the decay rate of the learning rate schedule (default: %(default)s)')
    parser.add_argument('--batchnorm', type=float, default=0,
                        help='if nonzero, batchnormalize each layer with the given momentum (default: %(default)s)')
    parser.add_argument('--training_set_size', type=int, default=20000,
                        help='if nonzero, batchnormalize each layer with the given momentum (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=56,
                        help='if nonzero, batchnormalize each layer with the given momentum (default: %(default)s)')

    args = parser.parse_args()
    
    qeccode = Steane_FT(args.prob[0], args.prob[1], args.prob[2], args.cir_id)
    in_dim = 2*qeccode.num_anc
    out_dim = 2*qeccode.num_data

    model = create_model(in_dim=in_dim, out_dim=out_dim,
                         number_of_samples = args.onthefly[0],
                         hidden_sizes=args.layers,
                         hidden_act=args.hact,
                         act=args.act,
                         loss=args.loss,
                         learning_rate=args.learningrate,
                         batchnorm=args.batchnorm,
                         decayrate=args.decayrate)

    if args.epochs:
        ds = read_data('../datasets/112000000cir_id2per0.001\dataset.txt', 24,args.training_set_size)
        dat = MyBatchGenerator(ds,args.batch)

        hist = model.fit_generator(dat, steps_per_epoch=2,
                                   epochs = args.epochs,
                                   verbose=2,
                                   use_multiprocessing=True,
                                   workers=args.workers,
                                   max_queue_size=args.workers)

        with open('../cir_id'+str(args.cir_id)+ '/'+args.out+'data.log', 'w') as f:
            f.write(str(args))
            f.write('\n')
            f.write(str((hist.params, hist.history)))
        model.save('../cir_id'+str(args.cir_id)+'/'+args.out+'nn.model')


    if args.eval:
        model = tf.keras.models.load_model('nn_model/'+args.out+'.model')
        dat = data_generator(qeccode, args.batch)
        val = data_generator(qeccode, args.batch)
#        epoch_callbacks = training_procedure.history
        hist = model.fit_generator(dat, args.onthefly[0]//args.batch, 1,
                                   validation_data=val, validation_steps=args.onthefly[1]//args.batch,verbose=2)

#        err = check(outputname=args.out, trials=args.onthefly[1], qeccode=qeccode)
        print(err/args.onthefly[1])



