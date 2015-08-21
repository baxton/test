import gc
import re
import sys
import numpy as np
from array import array

from sklearn.linear_model import SGDClassifier

#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from ann import ANN

from utils import *


RATE = 500


EPOCHES = 200   #8000
MINI_BATCH_SIZE = 1

MIN = None
MAX = None


ZEROS_II = np.random.randint(0, 7062-6, 200)

FF = [ 296, 275, 64, 376, 455, 172, 309, 290, 480, 494, 264, 477, 102, 199, 269, 358, 181, 323, 74, 5, 423, 361, 35, 246, 149, 231, 352, 154, 68, 7, 270, 354, ]



TRAIN_POS = 0
TRAIN_NEG = 0
TEST_POS = 0
TEST_NEG = 0


def sigmoid(v):
    return 1. / (1. + np.exp(-v))




def get_shape(fname):
    m = re.match("^.*_([0-9]+)x([0-9]+)\.bin", fname).groups()
    return (int(m[0]), int(m[1]))




def read_data(fname, y_ch, data, events): 
    rows, cols = get_shape(fname)

    pos_prob_to_stay = .7;

    max_rows = data.shape[0]
    cur_pos = 0;
    step = 1000;
    rows_processed = 0

    tmp = None

    with open(fname, "rb") as fin:
        r = 0
	while r < rows:
            if r + step > rows:
                step = rows - r
            r += step

            tmp = np.fromfile(fin, dtype=np.float64, count=step*cols, sep='')
            tmp = tmp.reshape((step, cols))


            for i in range(step):
                if cur_pos < max_rows:
                    data[cur_pos] = tmp[i,:-6]
                    events[cur_pos] = tmp[i,-6+y_ch]
                    cur_pos += 1
                else:
                    p = np.random.randint(0, max_rows)
#                    if p >= max_rows and events[p] == 1:
#                        p = np.random.randint(0, max_rows)
                    if p < max_rows:
                        e = tmp[i,-6+y_ch]
                        if e == 1. or \
                           e == 0 and events[p] == 1 and pos_prob_to_stay < np.random.rand(1) or \
                           e == 0 and events[p] == 0:
                            data[p] = tmp[i,:-6]
                            events[p] = e
                rows_processed += 1
                    

    return cur_pos
    




def read_file(fname, y_ch, all=False, step=5000):

    rows, cols = get_shape(fname)

    pos_p = .05
    neg_p = .01

    data = None


    with open(fname, "rb") as fin:
        r = 0
	while r < rows:
            if r + step > rows:
                step = rows - r

            r += step
            if None != data:
                del data
                gc.collect()
            #gc.collect()
            data = np.fromfile(fin, dtype=np.float64, count=step*cols, sep='')
            data = data.reshape((step, cols))


            for i in range(step):
#                for c in range(cols):
#                    if np.isnan(data[i,c]) or np.isinf(data[i,c]):
#                        print "NAN/INF", r, i, c
#                        print data[i]
                if not all:
                    p = np.random.rand(1)
                    if data[i,-6 + y_ch] == 0:
                        if p <= neg_p:
                            continue
                    else:
                        if p <= pos_p:
                            continue
                events   = data[i,-6:]
                data_row = data[i,:-6]

                yield data_row, events







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




def get_stat(fname):
    fn = fname + ".stat"
    print "read stat from", fn
    cur_state = ""
    vmean = None
    vmin = None
    vmax = None
    with open(fn, "r") as fin:
        for line in fin:
            line = line.strip()
            if line == "# Mean":
                cur_state = "mean"
            elif line == "# Min":
                cur_state = "min"
            elif line == "# Max":
                cur_state = "max"
            else:
                if cur_state == "mean":
                    vmean = eval(line)
                    cur_state = ""
                elif cur_state == "min":
                    vmin = eval(line)
                    cur_state = ""
                elif cur_state == "max":
                    vmax = eval(line)
                    cur_state = ""
                else:
                    pass
    return np.array(vmean, dtype=np.float64), np.array(vmin, dtype=np.float64), np.array(vmax, dtype=np.float64)






def train(fn_train, fn_test, y_ch):
    shape = get_shape(fn_train)

    MAX_ROWS = 5000

    #c_t0 = RandomForestClassifier(200)  #, max_depth=3)
    c_t0 = GradientBoostingClassifier(n_estimators=800)


    N = shape[1] - 6


    data = np.zeros((MAX_ROWS, N), dtype=np.float64)
    events = np.zeros((MAX_ROWS, 1), dtype=np.float64)


    read_rows = read_data(fn_train, y_ch, data, events)

    div = (data.max(axis=0) - data.min(axis=0))
    div[div==0.] = 0.00000001
    data /= div

    data = data[:,-496:][:,FF]

    c_t0.fit(data, events)

    print_importance(c_t0.feature_importances_) 

    test(c_t0, fn_test, div, y_ch)






def train_sgd(fn_train, fn_test, y_ch):
    shape = get_shape(fn_train)
    N = shape[1] - 6

    N = len(FF)

    MAX_ROWS = 2000

    loss = "log"
#    c_t0 = SGDClassifier(loss, n_iter=40)
    c_t0 = ANN([N, 1, 1])


    data = np.zeros((MAX_ROWS, shape[1]-6), dtype=np.float64)
    events = np.zeros((MAX_ROWS, 1), dtype=np.float64)


    read_rows = read_data(fn_train, y_ch, data, events)
    div = (data.max(axis=0) - data.min(axis=0))
    div[div==0.] = 0.00000001

    data /= div

    s = events.sum() 
    print "POS", s, "NEG", events.shape[0] - s

    exit = False


    data = data[:,-496:]  #[:,FF].copy()



    for e in range(EPOCHES):
        indices = range(read_rows)
        np.random.shuffle(indices)


        avr_cost = 0.
        alpha = 999
        for r in indices:   # range(read_rows):
            if type(c_t0) == ANN:
                out_par = []
                c_t0.partial_fit(data[r], events[r], [0, 1], out_par)
                avr_cost += c_t0.cost.value
                alpha = c_t0.alpha.value
                if out_par[0] < 0.0000001:
                    print "Alpha", alpha, "stopping"
                    exit = True
                    break
            else:
                c_t0.partial_fit(data[r], events[r], [0, 1])
        if exit:
            break

        avr_cost = avr_cost / len(indices)
        print "Epoch %d out of %d  done, avr cost %f, alpha %f" % (e, EPOCHES, avr_cost, alpha)

 #       if avr_cost < 0.00001:
 #           break

        if e > 0 and 0 == (e % 100):
            test(c_t0, fn_test, div, y_ch)

    test(c_t0, fn_test, div, y_ch)








def test(c_t0, fname, div, y_ch):

    y_true = []
    predictions = []

    global TEST_POS, TEST_NEG

    r = 0

    nan = float('nan')


    for data, events in read_file(fname, y_ch, all=True, step=1000):
        data /= div
        tmp = data

        tmp = tmp[-496:]   ##[FF]


        y_true.append(events[y_ch])

        if events[y_ch] == 1:
            TEST_POS += 1
        else:
            TEST_NEG += 1

        p_t0 = c_t0.predict_proba([tmp])


        p_t0 = p_t0[0,1]
	predictions.append(p_t0)


        r += 1
        if 0 == (r % 3000):
            print "Step %d" % (r,)
            if np.unique(y_true).shape[0] > 1:
                print "intermed auc", roc_auc_score(y_true, predictions)



    auc = roc_auc_score(y_true, predictions)
    print "AUC", auc


    


def main():
    np.random.seed()

    fn_train = sys.argv[1]
    fn_test = sys.argv[2]
    y_ch = int(sys.argv[3])


    train_sgd(fn_train, fn_test, y_ch)
#    train(fn_train, fn_test, y_ch)




    print "POS:", TRAIN_POS
    print "NEG:", TRAIN_NEG
    print "Test POS:", TEST_POS
    print "Test NEG:", TEST_NEG


main()
