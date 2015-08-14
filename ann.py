
import os
import sys
import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import ctypes

import ann_py

ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\ann\ann.dll")



def plotit(X, Y, c='b', m='x', clear=True, show=True, subplot=None):
    if clear:
        plt.clf()
    if None != subplot:
        plt.subplot(*subplot)
    plt.scatter(X, Y, c=c, marker=m)
    if show:
        plt.show()


def gen_data(N, disp=False, subplot=(1,1,1)):

    np.random.seed()

    CX = np.array([0, 10, 8])
    CY = np.array([0, 10, -7])

    mul = 3.

    X1 = [d*mul + CX[0] for d in np.random.normal(size=N)]
    Y1 = [d*mul + CY[0] for d in np.random.normal(size=N)]

    X2 = [d*mul + CX[1] for d in np.random.normal(size=N/2)]
    Y2 = [d*mul + CY[1] for d in np.random.normal(size=N/2)]

    X3 = [d*mul + CX[2] for d in np.random.normal(size=9)]
    Y3 = [d*mul + CY[2] for d in np.random.normal(size=9)]


    if disp:
        plotit(CX, CY, c='b', m='x', clear=False,  show=False, subplot=subplot)
        plotit(X1, Y1, c='r', m='o', clear=False, show=False, subplot=subplot)
        plotit(X2, Y2, c='y', m='o', clear=False, show=False, subplot=subplot)
        plotit(X3, Y3, c='g', m='o', clear=False, show=False,  subplot=subplot)

    return sp.array(zip(X1, Y1)), sp.array(zip(X2, Y2)), sp.array(zip(X3, Y3))




def train(ds1, ds2, ds3):
    X = sp.concatenate((ds1, ds2, ds3), axis=0)
    #Y = sp.concatenate(( [[1,0,0]]*ds1.shape[0], [[0,1,0]]*ds2.shape[0], [[0,0,1]]*ds3.shape[0] ))
    Y = sp.concatenate(( [0]*ds1.shape[0], [1]*ds2.shape[0], [2]*ds3.shape[0] ))

    ann = RandomForestClassifier()
    ann.fit(X, Y)

    return ann

def train_ann(ds1, ds2, ds3):


    #np.random.shuffle(indices)

##    X = np.array([[0,0]] * N, dtype=np.float32)
##    Y = np.array([[0,0,0]] * N, dtype=np.float32)

    X = sp.concatenate((ds1, ds2, ds3), axis=0).astype(np.float32)
    Y = sp.concatenate(( [[1,0,0]]*ds1.shape[0], [[0,1,0]]*(ds2.shape[0]), [[0,0,1]]*(ds3.shape[0]) )).astype(np.float32)

    N = ds1.shape[0] + ds2.shape[0] + ds3.shape[0]

    indices = np.array(range(N), dtype=int)


    ann = ANN_DLL.ann_create()

    alpha = ctypes.c_float(.08)

    #MBS = X.shape[0]
    MBS = 10
    for cnt in range(1, 6000):
        tmpIndices = np.random.choice(indices, MBS, replace=False)
        tmpX = X[tmpIndices].astype(np.float32)
        tmpY = Y[tmpIndices].astype(np.float32)

        ANN_DLL.ann_fit(ctypes.c_void_p(ann), tmpX.ctypes.data, tmpY.ctypes.data, ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_float(0), ctypes.c_int(5))
        alpha.value = .08

    return ann


def train_ann_py(ds1, ds2, ds3):

    X = sp.concatenate((ds1, ds2, ds3, ds3, ds3, ds3, ds3, ds3), axis=0).astype(np.float32)
    Y = sp.concatenate(( [[1,0,0]]*ds1.shape[0], [[0,1,0]]*(ds2.shape[0]), [[0,0,1]]*(6*ds3.shape[0]) )).astype(np.float32)

    N = ds1.shape[0] + ds2.shape[0] + 6*ds3.shape[0]

    #MBS = X.shape[0]
    MBS = 10

    indices = np.array(range(N), dtype=int)

    ann = ann_py.ANN(MBS, activation=ann_py.f_sigmoid)

    alpha = .08

    for cnt in range(1, 6000):
        tmpIndices = np.random.choice(indices, MBS, replace=False)
        tmpX = X[tmpIndices]
        tmpY = Y[tmpIndices]

        ann.fit_minibatch(tmpX, tmpY, alpha)
        alpha = .08

    return ann



def test(rf, ann, anp):

    ds1, ds2, ds3 = gen_data(100, disp=True, subplot=(5,1,2))



    ## sklearn RF
    for i in range(ds1.shape[0]):
        x = ds1[i]
        p = rf.predict(x)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds1[i,0], ds1[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,3))

    for i in range(ds2.shape[0]):
        x = ds2[i]
        p = rf.predict(x)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds2[i,0], ds2[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,3))

    for i in range(ds3.shape[0]):
        x = ds3[i]
        p = rf.predict(x)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds3[i,0], ds3[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,3))


    ## ANN PY

    p1 = 200. / 309
    p2 = 100. / 309
    p3 = 9. / 309

    for i in range(ds1.shape[0]):
        x = ds1[i]
        predictions = anp.forward_propagate(x.reshape((1,x.shape[0])))
        predictions = predictions.reshape((3,))
##        predictions[0] *= p1
##        predictions[1] *= p2
##        predictions[2] *= p3
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds1[i,0], ds1[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,4))

    for i in range(ds2.shape[0]):
        x = ds2[i]
        predictions = anp.forward_propagate(x.reshape((1,x.shape[0])))
        predictions = predictions.reshape((3,))
##        predictions[0] *= p1
##        predictions[1] *= p2
##        predictions[2] *= p3
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds2[i,0], ds2[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,4))

    for i in range(ds3.shape[0]):
        x = ds3[i]
        predictions = anp.forward_propagate(x.reshape((1,x.shape[0])))
        predictions = predictions.reshape((3,))
##        predictions[0] *= p1
##        predictions[1] *= p2
##        predictions[2] *= p3
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds3[i,0], ds3[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,4))



    ## ANN

    p1 = 200. / 309
    p2 = 100. / 309
    p3 = 9. / 309

    for i in range(ds1.shape[0]):
        x = ds1[i].astype(np.float32)
        predictions = np.array([0] * 3, dtype=np.float32)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, predictions.ctypes.data, ctypes.c_int(1))
##        predictions[0] *= p1
##        predictions[1] *= p2
##        predictions[2] *= p3
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds1[i,0], ds1[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,5))

    for i in range(ds2.shape[0]):
        x = ds2[i].astype(np.float32)
        predictions = np.array([0] * 3, dtype=np.float32)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, predictions.ctypes.data, ctypes.c_int(1))
##        predictions[0] *= p1
##        predictions[1] *= p2
##        predictions[2] *= p3
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds2[i,0], ds2[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,5))

    for i in range(ds3.shape[0]):
        x = ds3[i].astype(np.float32)
        predictions = np.array([0] * 3, dtype=np.float32)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, predictions.ctypes.data, ctypes.c_int(1))
##        predictions[0] *= p1
##        predictions[1] *= p2
##        predictions[2] *= p3
        p = np.argmax(predictions)
        c = 'r' if p == 0 else 'b' if p == 1 else 'g'
        m = 'x' if p == 0 else 'o' if p == 1 else 's'
        plotit(ds3[i,0], ds3[i,1], c=c, m=m, clear=False, show=False, subplot=(5,1,5))


    ##



def main():
    plt.clf()

    sp.random.seed()
    ds1, ds2, ds3 = gen_data(200, disp=True, subplot=(5,1,1))
    rf = train(ds1, ds2, ds3)
    ann = train_ann(ds1, ds2, ds3)
    anp = train_ann_py(ds1, ds2, ds3)
    test(rf, ann, anp)

    plt.show()




if __name__ == '__main__':
    main()
