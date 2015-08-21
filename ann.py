


import os
import sys
import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import ctypes

#import ann_py

#ANN_DLL = ctypes.cdll.LoadLibrary(r"/home/maxim/kaggle/ann/libann.so")
ANN_DLL = ctypes.cdll.LoadLibrary(r"c:\\temp\\test_python\\ann\\ann2.dll")




class ANN(object):
    def __init__(self, sizes):
        self.ss = np.array(sizes, dtype=np.int32)
        self.ann = ANN_DLL.ann_create(self.ss.ctypes.data, ctypes.c_int(self.ss.shape[0]))
        self.alpha = ctypes.c_double(.0001)
        self.cost = ctypes.c_double(0.)


    def partial_fit(self, X, Y, dummy, out_params=None):
        R = X.shape[0] if len(X.shape) == 2 else 1
        ANN_DLL.ann_fit(ctypes.c_void_p(self.ann), X.ctypes.data, Y.ctypes.data, ctypes.c_int(R), ctypes.addressof(self.alpha), ctypes.c_double(16), ctypes.c_int(1), ctypes.addressof(self.cost))
        if None != out_params:
            out_params.append(self.alpha.value)
            out_params.append(self.cost)


    def predict_proba(self, X):
        if type(X) == list:
            X = np.array(X, dtype=np.float64)
        R = X.shape[0] if len(X.shape) == 2 else 1
        C = self.ss[-1]
        predictions = np.array([0] * R * C, dtype=np.float64)
        ANN_DLL.ann_predict(ctypes.c_void_p(self.ann), X.ctypes.data, predictions.ctypes.data, ctypes.c_int(R))

        predictions = predictions.reshape((R, C))

        if C == 1:
            res = np.zeros((R, 2), dtype=np.float64)
            for i,v in enumerate(predictions):
                res[i,0] = 1. - v[0]
                res[i,1] = v[0]
            return res

        return predictions




