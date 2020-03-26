from flagbridgeqec.circuits import esmx_anc, esmz_anc, esmxs_anc3, esmzs_anc3, esmxs_anc4, esmzs_anc4


def esm2(idling=False):
    """                                                   
    syndrome extractor circuit for steane circuit 2, two ancillas per face, 6 
    checks in series      
    """
    cirs = []
    cirs.append(esmx_anc([4,5,6,7], [12, 13], db=3, idling=idling))
    cirs.append(esmx_anc([1,2,4,5], [8, 9], idling=idling))
    cirs.append(esmx_anc([1,3,4,7], [10, 11], idling=idling))
    cirs.append(esmz_anc([4,5,6,7], [120, 130],  db=3, idling=idling))
    cirs.append(esmz_anc([1,2,4,5], [80, 90], idling=idling))
    cirs.append(esmz_anc([1,3,4,7], [100, 110], idling=idling))
    return cirs

def esm3(idling=False):
    """
    syndrome extractor circuit for steane circuit 2, two ancillas per face, 
    6 checks in series      
    """
    cirs = []
    cirs.append(esmx_anc([4,5,6,7], [12, 13], idling=idling, db=3))
    cirs.append(esmx_anc([1,2,4,5], [8, 9], idling=idling))
    cirs.append(esmx_anc([1,3,4,7], [10, 11], idling=idling))
    cirs.append(esmz_anc([4,5,6,7], [120, 130], idling=idling, db=3))
    cirs.append(esmz_anc([1,2,4,5], [80, 90], idling=idling))
    cirs.append(esmz_anc([1,3,4,7], [100, 110], idling=idling))
    return cirs

def esm4(idling=False):
    """                                                                   
    syndrome extractor circuit for steane circuit 5, two checks in parallel  
    """
    cirs = []
    cirs.append(esmxs_anc3( [3,7,2,5], [3,7,1,4], [8, 10, 9], idling=idling))
    cirs.append(esmx_anc([4,5,6,7], [12, 13, 14], idling=idling))
    cirs.append(esmzs_anc3( [3,7,2,5], [3,7,1,4], [80, 100, 90], idling=idling))
    cirs.append(esmz_anc([4,5,6,7], [120, 130, 140], idling=idling))
    return cirs

def esm5(idling=False):
    """              
    syndrome extractor circuit for steane circuit 5, two checks in parallel   
    """
    cirs = []
    cirs.append(esmxs_anc3( [3,7,2,5], [3,7,1,4], [8, 10, 9], idling=idling))
    cirs.append(esmx_anc([4,5,6,7], [13, 12], idling=idling))
    cirs.append(esmzs_anc3( [3,7,2,5], [3,7,1,4], [80, 100, 90], idling=idling))
    cirs.append(esmz_anc([4,5,6,7], [120, 130], idling=idling))
    return cirs

def esm7(idling=False):
    """        
    syndrome extractor circuit for steane circuit 5, two checks in parallel  
    """
    cirs = []
    cirs.append(esmxs_anc4([4,1,5,2], [4,1,3,7], [4,5,6,7], [8, 10, 12, 9],
                              chao=False, idling=idling))
    cirs.append(esmzs_anc4([4,1,5,2], [4,1,3,7], [4,5,6,7], 
                              [80, 100, 120, 90], chao=False, idling=idling))

    return cirs

if __name__ == '__main__':
    circuit = esm2()
