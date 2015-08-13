
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

WIDTH = 512
STEP = 10


def load(subj, ser):
    fn_events = "subj%d_series%d_events.csv.bin" % (subj, ser)
    fn_data   = "subj%d_series%d_data.csv.bin" % (subj, ser)

    events = load_raw_data(train_path + "subj1_series1_events.csv.bin")
    data = load_raw_data(train_path + "subj1_series1_data.csv.bin")

    events = events.reshape((CHANNELS_EVENTS, events.shape[0] / CHANNELS_EVENTS))
    data = data.reshape((CHANNELS_DATA, data.shape[0] / CHANNELS_DATA))

    return data, events






def prep_features(subjs, sers, x_ch, y_ch, fn_pref="data"):
    rows = 0
    cols = 0

    fn = "./%s_%dx%d.bin" % (fn_pref, rows, cols)
    with open(fn, "wb+") as fout:
        for s in range(len(subjs)):
            data, events = load(subjs[s], sers[s])
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
       
    fn_new = "./%s_%dx%d.bin" % (fn_pref, rows, cols)
    os.rename(fn, fn_new) 
    print rows, cols
    
                
    
def prep_for_testing(subj, ser, x_ch, y_ch, fn_pref="test"):
    rows = 0
    cols = 0

    fn = "./%s_test_%dx%d.bin" % (fn_pref, rows, cols)
    with open(fn, "wb+") as fout:
        data, events = load(subj, ser)
        data = data[x_ch,:]
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

    fn_new = "./%s_test_%dx%d.bin" % (fn_pref, rows, cols)
    os.rename(fn, fn_new)
    print rows, cols




def main():
    fn_pref = "data"
    if len(sys.argv) > 1:
        fn_pref = sys.argv[1]
    prep_features([1,3, 5], [5,6,2], x_ch=5, y_ch=5, fn_pref=fn_pref)
    prep_for_testing(7, 2, x_ch=5, y_ch=5, fn_pref=fn_pref)


main()
