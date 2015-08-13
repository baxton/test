
import re
import sys
import numpy as np
from array import array

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from utils import *


RATE = 500

CHANNELS_EVENTS = 6
CHANNELS_DATA = 32

WIDTH = 512
STEP = 10

ROWS = 11899
COLS = 1201

FNAME = ""

EPOCHES = 1000
MINI_BATCH_SIZE = 500

MIN = 0
MAX = 0


def load_test(fname):
    m = re.match("^.*_([0-9]+)x([0-9]+)\.bin", fname).groups()
    rows = int(m[0])
    cols = int(m[1])

    with open(fname, "rb") as fin:
        data = np.fromfile(fin, dtype=np.float64, sep='')

    events = data[:,-1].reshape((rows,))
    data = data[:,:-1].reshape((rows, cols-1))

    return data, events


def read_test(fname):
    m = re.match("^.*_([0-9]+)x([0-9]+)\.bin", fname).groups()
    rows = int(m[0])
    cols = int(m[1])

    with open(fname, "rb") as fin:
        try:
            while True:
                data = np.fromfile(fin, dtype=np.float64, count=cols, sep='')

                events = data[-1]
                data = data[:-1]

                yield data, events
        except:
            pass






def load(fname):
    global ROWS, COLS
    m = re.match("^.*_([0-9]+)x([0-9]+)\.bin", fname).groups()
    ROWS = int(m[0])
    COLS = int(m[1])
    data = np.fromfile(fname, dtype=np.float64, sep='')
    return data.reshape((ROWS, COLS))   



def print_importance(a):
    print "=== Features importance"
    for v in a:
        print a
    print "=== End"



def train():
    data = load(FNAME)

    global MIN, MAX
    MIN = data.min(axis=0)[:-3]
    MAX = data.max(axis=0)[:-3]
    MIN[MIN==MAX] = 0.

    div = (MAX - MIN)
    div[div==0.] = 0.00000001
    data[:,:-3] /= div


    loss = "log"
    #loss = "modified_huber"
    c_t0 = RandomForestClassifier()
    c_t1 = RandomForestClassifier()
    c_t2 = RandomForestClassifier()

    c_t0.fit(data[:,:-3], data[:,-3])
    c_t1.fit(data[:,:-3], data[:,-2])
    c_t2.fit(data[:,:-3], data[:,-1])

    data = None

    print_importance(c_t0.feature_importances_) 
    print_importance(c_t1.feature_importances_) 
    print_importance(c_t2.feature_importances_) 


    return c_t0, c_t1, c_t2



def train_sgd():
    data = load(FNAME)

    global MIN, MAX
    MIN = data.min(axis=0)[:-3]
    MAX = data.max(axis=0)[:-3]
    MIN[MIN==MAX] = 0.

    div = (MAX - MIN)
    div[div==0.] = 0.00000001
    data[:,:-3] /= div


    loss = "log"
    #loss = "modified_huber"
    c_t0 = SGDClassifier(loss, n_iter=10)
    c_t1 = SGDClassifier(loss, n_iter=10)
    c_t2 = SGDClassifier(loss, n_iter=10)

    indices = range(ROWS)

    for e in range(EPOCHES):
        np.random.shuffle(indices)
        for_train = indices[:MINI_BATCH_SIZE]

        c_t0.partial_fit(data[for_train,:-3], data[for_train,-3], [0, 1])
        c_t1.partial_fit(data[for_train,:-3], data[for_train,-2], [0, 1])
        c_t2.partial_fit(data[for_train,:-3], data[for_train,-1], [0, 1])

        print "Epoch %d out of %d  done" % (e, EPOCHES)

    data = None

    return c_t0, c_t1, c_t2


def test(c_t0, c_t1, c_t2, fname):

    div = (MAX - MIN)
    div[div==0.] = 0.00000001

    y_true = []
    predictions = []

    r = 0

    for data, event in read_test(fname):
        tmp = data
        tmp /= div

        y_true.append(event)

        p_t0 = c_t0.predict_proba([tmp])[0,1]
        p_t1 = c_t1.predict_proba([tmp])[0,1]
        p_t2 = c_t2.predict_proba([tmp])[0,1]

        if 0 == len(predictions):
            # step 0
            predictions.append(p_t0)
            predictions.append(p_t1)
            predictions.append(p_t2)
        elif 3 == len(predictions):
            # step 1
            predictions[-2] = (predictions[-2] + p_t0) / 2.
            predictions[-1] = (predictions[-1] + p_t1)
            predictions.append(p_t2)
        else:
            # step n
            predictions[-2] = (predictions[-2] + p_t0) / 3.
            predictions[-1] = (predictions[-1] + p_t1)
            predictions.append(p_t2)

        r += 1
        if 0 == (r % 1500):
            print "Step %d" % (r,)
            print "intermed auc", roc_auc_score(y_true, predictions[:-2])

    predictions = predictions[:-2]

    auc = roc_auc_score(y_true, predictions)
    print "AUC", auc


    


def main():
    np.random.seed()

    global FNAME
    FNAME = sys.argv[1]
    fname_test = sys.argv[2]

    #c_t0, c_t1, c_t2 = train_sgd()
    c_t0, c_t1, c_t2 = train()
    test(c_t0, c_t1, c_t2, fname_test)


main()
