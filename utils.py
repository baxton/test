import numpy as np
from array import array

base_path = '../'
data_path = base_path + 'data/'
train_path = data_path + 'train/'
test_path = data_path + 'test/'


WIDTH = 512*4



def load_raw_data(fname):
    with open(fname, "rb") as fin:
        fin.seek(0,2)
        file_size = fin.tell()
        fin.seek(0)

        a = array('f')
        a.fromfile(fin, file_size / 4)
    return np.array(a, dtype=np.float32)


def features(data, width, rate):

#    w = np.hamming(data.shape[0])
#    data *= w

    F = np.fft.fft(data)

    idx = int(40. * width / rate)
    F = F[:idx]
    N = F.shape[0]

    A = np.sqrt(F.real**2 + F.imag**2) / N
    AL = np.log(1. + A)

    P  = F.imag
    P2 = P**2
    P2L = np.log(1. + P2)

    tmp = F.real
    tmp[tmp==0] = 0.0000001
    PA = np.arctan(P / tmp)
    PAL = np.log(1. + np.abs(PA))
    

    F = F.real

    bins = np.linspace(F.min(), F.max(), 5)
    tmp = np.histogram(F, bins)[0].astype(np.float64)
    tmp[tmp==0] = 0.0000001
    FM = np.histogram(F, bins, weights=F)[0] / tmp
    
    F2 = F**2
    bins = np.linspace(F2.min(), F2.max(), 5)
    FSS = np.histogram(F2, bins, weights=F2)[0]

    bins = np.linspace(data.min(), data.max(), 5)
    tmp = np.histogram(data, bins)[0].astype(np.float64)
    tmp[tmp==0.] = 0.0000001
    DM = np.histogram(data, bins, weights=data)[0] / tmp
    
    m = data.mean()
    v = data.var()
    s = data.std()

    N = data.shape[0]
    R = np.correlate(data, data, mode='full')[-N:] / (v * np.arange(N, 0, -1))

    D2 = data**2
    #D2[D2 == 0.] = 0.0000001
    D2L = np.log(1. + D2)


    result = np.concatenate((F, F2, FM, FSS, A, AL, P, P2, P2L, PA, PAL, DM, D2, D2L, R, [m, v, s],))
    return array('d', result)
    #return np.array(result, dtype=np.float64)



