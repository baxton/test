
import re
import sys
import numpy as np
from array import array

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from ann import ANN

from utils import *


RATE = 500

CHANNELS_EVENTS = 6
CHANNELS_DATA = 32

STEP = 10

ROWS = 11899
COLS = 1201

FNAME = ""

EPOCHES = 5
MINI_BATCH_SIZE = 500

MIN = 0
MAX = 0



TRAIN_POS = 0
TRAIN_NEG = 0
TEST_POS = 0
TEST_NEG = 0


def sigmoid(v):
    return 1. / (1. + np.exp(-v))


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

                event = data[-1]
                data = data[:-1]

                yield data, event
        except:
            pass






def load(fname):
    global ROWS, COLS
    m = re.match("^.*_([0-9]+)x([0-9]+)\.bin", fname).groups()
    ROWS = int(m[0])
    COLS = int(m[1])
    data = np.fromfile(fname, dtype=np.float64, sep='')
    return data.reshape((ROWS, COLS))   





def morfo(a):
    N = len(a)
    for i in range(2, N-2):
        if a[i] > 1.:
            a[i] = 1.
        elif a[i] < .1:
            a[i] = 0.
        elif a[i-2] >= .4 and \
           a[i-1] >= .4 and \
           a[i]   >= .3 and \
           a[i+1] >= .4 and \
           a[i+2] >= .4:
            a[i] = 1.
        elif a[i-2] >= .5 and \
           a[i-1] >= .5 and \
           a[i]   >= .0 and \
           a[i+1] >= .5 and \
           a[i+2] >= .5:
            a[i] += .1
        elif a[i-2] == 0. and \
             a[i-1] == 0. and \
             a[i]   <= .3 and \
             a[i+1] == 0. and \
             a[i+2] == 0.:
            a[i] = 0.




def print_importance(a):
    indices = np.argsort(a)
    print "=== Features importance"
    for i, v in zip(indices, a[indices]):
        print i, v
    print "=== End"
    N = indices.shape[0]
    print "[",
    for i in range(1, N/2+1):
        print "%d," % indices[-i],
    print "]"



def train():
    data = load(FNAME)
    print "data loaded"

    global MIN, MAX
    MIN = data.min(axis=0)[:-3]
    MAX = data.max(axis=0)[:-3]
    MIN[MIN==MAX] = 0.

    div = (MAX - MIN)
    div[div==0.] = 0.00000001
    data[:,:-3] /= div

    data[:,-3:] -= .5





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
#    print_importance(c_t1.feature_importances_) 
#    print_importance(c_t2.feature_importances_) 


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


    global TRAIN_POS, TRAIN_NEG
    TRAIN_POS = (data[:,-3] == 1).sum()
    TRAIN_NEG = (data[:,-3] == 0).sum()


    loss = "log"
    #loss = "modified_huber"
#    c_t0 = SGDClassifier(loss, n_iter=5)
#    c_t1 = SGDClassifier(loss, n_iter=5)
#    c_t2 = SGDClassifier(loss, n_iter=5)

    c_t0 = ANN([data.shape[1] - 3, 2000, 1])
#    c_t1 = ANN([data.shape[1] - 3, 100, 2])
#    c_t2 = ANN([data.shape[1] - 3, 100, 2])


    N = data.shape[0]
    indices = np.array(range(N))
    pos = data[:,-3] == 1
    neg = ~pos

    pos_ii = indices[pos]
    neg_ii = indices[neg]


    for e in range(EPOCHES):
        np.random.shuffle(pos_ii)
        np.random.shuffle(neg_ii)
        for_train = np.concatenate((pos_ii[:MINI_BATCH_SIZE], neg_ii[:MINI_BATCH_SIZE+1]))

#        y0 = np.array([1. - data[for_train,-3], data[for_train,-3]], dtype=np.float64)
#        y1 = np.array([1. - data[for_train,-2], data[for_train,-2]], dtype=np.float64)
#        y2 = np.array([1. - data[for_train,-1], data[for_train,-1]], dtype=np.float64)

        y0 = data[for_train,-3]
#        y1 = data[for_train,-2]
#        y2 = data[for_train,-1]

        c_t0.partial_fit(data[for_train,:-3], y0, [0, 1])
#        c_t1.partial_fit(data[for_train,:-3], y1, [0, 1])
#        c_t2.partial_fit(data[for_train,:-3], y2, [0, 1])

        print "Epoch %d out of %d  done" % (e, EPOCHES)

    data = None

    return c_t0, None, None  #, c_t1, c_t2


def test(c_t0, c_t1, c_t2, fname):

    div = (MAX - MIN)
    div[div==0.] = 0.00000001

    y_true = []
    predictions = []

    global TEST_POS, TEST_NEG

    r = 0


    for data, event in read_test(fname):
        tmp = data
        tmp /= div


        y_true.append(event)

        if event == 1:
            TEST_POS += 1
        else:
            TEST_NEG += 1

        p_t0 = c_t0.predict_proba([tmp])
#        print event, p_t0
#        p_t1 = c_t1.predict_proba([tmp])[0,1]
#        p_t2 = c_t2.predict_proba([tmp])[0,1]


        p_t0 = p_t0[0,1] / (p_t0[0,0] + p_t0[0,1]);
	predictions.append(p_t0)


#        if 0 == len(predictions):
#            # step 0
#            predictions.append(p_t0)
#            predictions.append(p_t1)
#            predictions.append(p_t2)
#        elif 3 == len(predictions):
#            # step 1
#            predictions[-2] = (predictions[-2] + p_t0) / 2.
#            predictions[-1] = (predictions[-1] + p_t1)
#            predictions.append(p_t2)
#        else:
#            # step n
#            predictions[-2] = (predictions[-2] + p_t0) / 3.
#            predictions[-1] = (predictions[-1] + p_t1)
#            predictions.append(p_t2)
#
#            #print event, predictions[-2]


        r += 1
        if 0 == (r % 1500):
            print "Step %d" % (r,)
            if np.unique(y_true).shape[0] > 1:
                print "intermed auc", roc_auc_score(y_true, predictions)

#    predictions = predictions[:-2]




#    print "=== After morfo"
#    for i in range(len(predictions)):
#        print y_true[i], predictions[i]

    auc = roc_auc_score(y_true, predictions)
    print "AUC", auc


    


def main():
    np.random.seed()

    global FNAME
    FNAME = sys.argv[1]
    fname_test = sys.argv[2]

    c_t0, c_t1, c_t2 = train_sgd()
    #c_t0, c_t1, c_t2 = train()
    test(c_t0, c_t1, c_t2, fname_test)

    print "POS:", TRAIN_POS
    print "NEG:", TRAIN_NEG
    print "Test POS:", TEST_POS
    print "Test NEG:", TEST_NEG


main()
