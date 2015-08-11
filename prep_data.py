
import sys
import numpy as np
from array import array







def save_array(src, file_name):
    with open(file_name, "wb+") as fout:
        src = src.T
        for i in range(src.shape[0]):
            a = array('f', src[i,:])
            a.tofile(fout)
     



def process(subj_no):
    for ser_no in range(1, 9):
        events_file = '../data/train/subj%d_series%d_events.csv' % (subj_no, ser_no)
        train_file = '../data/train/subj%d_series%d_data.csv' % (subj_no, ser_no)

        print "Processing: %d, %d" % (subj_no, ser_no)



        events = np.array([], dtype=np.float32).reshape((0, 6))
        data = np.array([], dtype=np.float32).reshape((0, 32))

        with open(events_file, "r") as fin:
            fin.readline()      # skip header
            row = None
            for line in fin:
                tokens = line.strip().split(",")
                row = np.array([float(v) for v in tokens[1:]], dtype=np.float32)
                events = np.concatenate((events, row.reshape((1, 6))), axis=0)
        save_array(events, events_file + ".bin")
        events = None
    

        with open(train_file, "r") as fin:
            fin.readline()      # skip header
            row = None
            for line in fin:
                tokens = line.strip().split(",")
                row = np.array([float(v) for v in tokens[1:]], dtype=np.float32)
                data = np.concatenate((data, row.reshape((1, 32))), axis=0)
        save_array(data, train_file + ".bin")
        data = None












def main():
    subj_no = int(sys.argv[1])
    process(subj_no)







main()
