import numpy as np

def read_data(fn,len_errs):
    """                                                                          
    Retrieve the .txt file that contains the dataset as obtained through sampling
    in the following format                                                      
    -------inputs---------    outputs   
    0 1 0 0 1 0 .... 0 0 0  523 ... 18  
    """
    ds = {}

    filename = './../datasets_optimized/' + str(fn)
    print(len_errs)
    with open(filename, 'r') as f:
        m = 0
        for line in f:
            m += 1

            data_smpl = line.split()
            
            syndrome = ''.join([d for d in data_smpl[:-(len_errs+1)]])
#            print(syndrome,'syndrome')
            data_dumb = np.array([float(d) for d in data_smpl[-(len_errs+1):-1]])
            ds[syndrome] = data_dumb

            if m % 1000 == 0:
                print(m, '\r', end='')
        f.close()
        print(m,'m')
        print(fn,'fn')
    return(ds)

def read_data_nn(fn,len_errs):
    """                                                                                                                                          Retrieve the .txt file that contains the dataset as obtained through sampling                                                                in the following format                                                                                                                      -------inputs---------    outputs                                                                                                            0 1 0 0 1 0 .... 0 0 0  523 ... 18                                                                                                           """
    ds = {}

    filename = '../../datasets/' + str(fn)
    print(len_errs)
    with open(filename, 'r') as f:
        m = 0
        for line in f:

            data_smpl = line.split()
            if data_smpl[-1] > 2:
                syndrome = ''.join([d for d in data_smpl[:-(len_errs+1)]])
                #            print(syndrome,'syndrome')                                                                                                       
                data_dumb = np.array([float(d) for d in data_smpl[-(len_errs+1):-1]])
                ds[syndrome] = data_dumb
                m += 1
            if m % 1000 == 0:
                print(m, '\r', end='')
        f.close()
        print(m,'m')
        print(fn,'fn')
    return(ds)

def read_data_perfect_lld(fn,len_errs):
    """                                                                                             
    Retrieve the .txt file that contains the dataset as obtained through sampling  
    in the following format                                                                         
    -------inputs---------    outputs                                                               
    0 1 0 0 1 0 .... 0 0 0  523 ... 18                                                              
    """
    ds = {}

    filename = './../datasets_optimized/' + str(fn)
    print(len_errs)
    with open(filename, 'r') as f:
        m = 0
        for line in f:
            m += 1

            data_smpl = line.split()

            syndrome = data_smpl[0]
            data_dumb = np.array(list(data_smpl[1]), dtype=int)
            ds[syndrome] = data_dumb

            if m % 1000 == 0:
                print(m, '\r', end='')
        f.close()
        print(m,'m')
        print(fn,'fn')
    return(ds)

