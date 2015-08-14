
import os
import sys
import numpy as np
from array import array

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from utils import *


RATE = 500

CHANNELS_EVENTS = 6
CHANNELS_DATA = 32

STEP = 100


def load(subj, ser, for_submission):
    events = None
    data = None

    if not for_submission:
        fn_events = "subj%d_series%d_events.csv.bin" % (subj, ser)
        fn_data   = "subj%d_series%d_data.csv.bin" % (subj, ser)

        events = load_raw_data(train_path + fn_events)
        data = load_raw_data(train_path + fn_data)

        events = events.reshape((CHANNELS_EVENTS, events.shape[0] / CHANNELS_EVENTS))
        data = data.reshape((CHANNELS_DATA, data.shape[0] / CHANNELS_DATA))
    else:
        fn_data   = "subj%d_series%d_data.csv.bin" % (subj, ser)
        data = load_raw_data(test_path + fn_data)
        data = data.reshape((CHANNELS_DATA, data.shape[0] / CHANNELS_DATA))


    return data, events






def prep_features(subjs, sers, x_ch, y_ch, fn_pref="data"):
    rows = 0
    cols = 0

    fn = "./%s_%d_%d_%dx%d.bin" % (fn_pref, x_ch, y_ch, rows, cols)
    with open(fn, "wb+") as fout:
        for s in range(len(subjs)):
            data, events = load(subjs[s], sers[s], for_submission=False)
            data = data[x_ch,:]
            events = events[y_ch,:]

            for beg in range(WIDTH-1, data.shape[0]-2, STEP):
                tmp = features(data[beg+1-WIDTH : beg+1], WIDTH, RATE)
                y = array('d', [events[beg], events[beg+1], events[beg+2]])
             
                tmp.tofile(fout)
                y.tofile(fout) 

                rows += 1
                cols = len(tmp) + 3

                if 0 == ((beg-WIDTH+1)/10 % 500): 
                    print "window %d out of %d done..." % (beg, data.shape[0]-1)
            print "subj %d done..." % s
       
    fn_new = "./%s_%d_%d_%dx%d.bin" % (fn_pref, x_ch, y_ch, rows, cols)
    os.rename(fn, fn_new) 
    print rows, cols
    
                
    
def prep_for_testing(subj, ser, x_ch, y_ch, fn_pref="test", for_submission=False):
    rows = 0
    cols = 0

    fn = "./%s_test_%d_%d_%dx%d.bin" % (fn_pref, x_ch, y_ch, rows, cols)
    with open(fn, "wb+") as fout:
        data, events = load(subj, ser, for_submission)
        data = data[x_ch,:]

        if not for_submission:
            events = events[y_ch,:]

            for beg in range(WIDTH-1, data.shape[0], 1):
                tmp = features(data[beg+1-WIDTH : beg+1], WIDTH, RATE)
                y = array('d', [events[beg]])

                tmp.tofile(fout)
                y.tofile(fout)

                rows += 1
                cols = len(tmp) + 1

                if 0 == ((beg-WIDTH+1)/10 % 500):
                    print "test wnd %d out of %d done..." % (beg, data.shape[0]-1)
        else:
            y = array('d', [0])
            for beg in range(WIDTH-1, data.shape[0], 1):
                tmp = features(data[beg+1-WIDTH : beg+1], WIDTH, RATE)

                tmp.tofile(fout)
                y.tofile(fout)

                rows += 1
                cols = len(tmp) + 1

                if 0 == ((beg-WIDTH+1)/10 % 500):
                    print "test wnd %d out of %d done..." % (beg, data.shape[0]-1)

    fn_new = "./%s_test_%d_%d_%dx%d.bin" % (fn_pref, x_ch, y_ch, rows, cols)
    os.rename(fn, fn_new)
    print rows, cols




def main():
    if len(sys.argv) == 1:
        print "Usege:"
        print "    $ python prep_data2.py <file_name_pref> <x ch> <y ch> <subj> <test ser> [--submission]"
        print "Example:"
        print "    $ python prep_data2.py TEST 12 3 1 1 --submission"
        return

    for_submission = False

    fn_pref = sys.argv[1]
    x_ch = int(sys.argv[2])
    y_ch = int(sys.argv[3])
    subj = int(sys.argv[4])
    test_ser = int(sys.argv[5])
    if 7 == len(sys.argv):
        for_submission = True


    tmp = range(1, 9)
    tmp.remove(test_ser)
    np.random.shuffle(tmp)
    ser_ii = tmp[:2]


    prep_features([subj, subj], ser_ii, x_ch=x_ch, y_ch=y_ch, fn_pref=fn_pref)
    prep_for_testing(subj, test_ser, x_ch=x_ch, y_ch=y_ch, fn_pref=fn_pref, for_submission=for_submission)

    print "-----------------------"
    print "File name prefix:", fn_pref
    print "X ch:", x_ch
    print "Y ch:", y_ch
    print "Test subj:", subj
    print "Test series:", test_ser
    print "Train subj:", subj
    print "Train series:", ser_ii
    print "For submission:", for_submission



main()
