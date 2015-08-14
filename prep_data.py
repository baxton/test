
import sys
import numpy as np
from array import array







def save_array(src, file_name):
    with open(file_name, "wb+") as fout:
        print len(src), len(src[1])
        R = len(src)
        for r in range(R):
            a = array('f', src[r])
            a.tofile(fout)
     



def process(subj_no):
    for ser_no in range(1, 9):
        events_file = '../data/train/subj%d_series%d_events.csv' % (subj_no, ser_no)
        train_file = '../data/train/subj%d_series%d_data.csv' % (subj_no, ser_no)

        print "Processing: %d, %d" % (subj_no, ser_no)



        events = []
        for i in range(6):
            events.append([])

        data = []
        for i in range(32):
            data.append([])

        cnt = 0

        with open(events_file, "r") as fin:
            fin.readline()      # skip header
            for line in fin:
                cnt += 1
                tokens = line.strip().split(",")
                for i in range(6):
                    events[i].append(float(tokens[i+1]))
        print "read", cnt, "rows"
        save_array(events, events_file + ".bin")
        events = None
    

        with open(train_file, "r") as fin:
            fin.readline()      # skip header
            for line in fin:
                tokens = line.strip().split(",")
                for i in range(32):
                    data[i].append(float(tokens[i+1]))
        save_array(data, train_file + ".bin")
        data = None
    # end of for 1-9







def process_test(subj_no):
    for ser_no in [9, 10]:
        train_file = '../data/test/subj%d_series%d_data.csv' % (subj_no, ser_no)

        print "Processing in test: %d, %d" % (subj_no, ser_no)


        data = []
        for i in range(32):
            data.append([])

        cnt = 0

        with open(train_file, "r") as fin:
            fin.readline()  
            for line in fin:
                tokens = line.strip().split(",")
                for i in range(32):
                    data[i].append(float(tokens[i+1]))
        save_array(data, train_file + ".bin")
        data = None
    # end of for 1-9







def main():
    subj_no = int(sys.argv[1])
    #process(subj_no)
    process_test(subj_no)







main()

