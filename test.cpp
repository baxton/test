


import os
import sys
import csv
import numpy as np
import scipy as sp
import dicom

from sklearn.linear_model import LogisticRegression



PATH_BASE = "..\\data\\"
PATH_CACHE = "..\\cache\\"



POSITIONS = {}


TRAIN_SET = []
TEST_SET  = []


BUFFERS = {}







def detect_rectangle(buffer, start_r, start_c, visited):
    R = buffer.shape[0]
    C = buffer.shape[1]

    top = start_r
    bottom = top

    left = start_c
    right = left
    
    S = 1

    key = (start_r, start_c)
    Q = [key]

    while 0 < len(Q):
        key = Q.pop()
        visited.add(key)

        # top
        if 0 < key[0]:
           r = key[0] - 1
           c = key[1]
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if top > r:
                       top = r
                   if bottom < r:
                       bottom = r
 
        # bottom
        if R > key[0] + 1:
           r = key[0] + 1
           c = key[1]
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if top > r:
                       top = r
                   if bottom < r:
                       bottom = r

        # left
        if 0 < key[1]:
           r = key[0]
           c = key[1] - 1
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if left > c:
                       left = c
                   if right < c:
                       right = c
 
        # right
        if C > key[1] + 1:
           r = key[0]
           c = key[1] + 1
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if left > c:
                       left = c
                   if right < c:
                       right = c
 
    return top, bottom, left, right, S







def get_rectangles(buffer):
    rects = []
    visited = set()

    R = buffer.shape[0]
    C = buffer.shape[1]

    for r in range(R):
        for c in range(C):
            if 0 == buffer[r,c]:
                continue

            key = (r,c)
            if key in visited:
                continue

            top, bottom, left, right, S = detect_rectangle(buffer, r, c, visited)
            rects.append((top, bottom, left, right, S))

    return rects







def cache(case_dir): 

    global BUFFERS

    dicom.VERBOSE = False

    scan_dirs = [d for d in os.listdir(PATH_BASE + case_dir + "\\study\\") if not d.startswith('.')]

    fnames = None

    # caching
    for d in scan_dirs:
        print "    %s" % d
        key = case_dir + "\\study\\" + d 
        if key in BUFFERS:
            break


        fname_cache = PATH_CACHE + "teach_heart_detector.py.%s.BUFFERS" % key.replace("\\", "_")
        if os.path.exists(fname_cache):
            with open(fname_cache, "r") as fin:
                text = fin.read()
                BUFFERS[key] = eval(text.replace("array", "np.array"))
                continue



        path = PATH_BASE + key + "\\"

        fnames = [f for f in os.listdir(path) if "dcm" in f and not f.startswith('.')]

        frames = []

        for f in fnames:
            print "        %s" % f
            fn = path + f
            file_buffer = np.fromfile(fn, dtype=np.uint8, sep='')
            dicom_obj = dicom.process(file_buffer)

            vec = np.array([
                       dicom_obj.Rows,
                       dicom_obj.Columns,
                       -1 if dicom_obj.Gender == "F" else 1,
                       dicom_obj.Age
                          ], dtype=np.float64)

            frames.append(dicom_obj.img_buffer)

        rects = process(frames)

        BUFFERS[key] = (rects, vec)
    
        with open(fname_cache, "w+") as fout:
            fout.write(str(BUFFERS[key]))
   
     




def process(frames):

    files_num = len(frames)

    prev = frames[0]
    shape2D = prev.shape
    freq = np.zeros(shape2D, dtype=float)

    cnt = 0
    thr = 2

    for i in range(1, files_num):
        cnt += 1
        if cnt != thr:
            continue

        cnt = 0
        curr = frames[i]
        #
        tmp = (prev - curr) ** 2
        mv = np.mean(tmp)
        tmp[tmp < (mv*4)] = 0
        freq[tmp != 0] += 1

        #
        prev = curr


    freq /= files_num
    freq *= 100




    for e in range(3):
        R = shape2D[0]
        C = shape2D[1]
        S = 6
        K = 12 + e
        res = np.zeros(shape2D, dtype=float)
        for r in range(R):
            if (r + S) >= R:
                continue
            for c in range(C):
                if (c + S) >= C:
                    continue
                tmp = freq[r : r + S, c : c + S]
                m = np.mean(tmp)
    
                #if K - 1 <= m and m < K + 1:
                if K < m:
                    res[r,c] = 100
                else:
                    res[r,c] = 0
    
        freq[S/2 :, S/2 :] = res[:-S/2, :-S/2]
        freq[:S/2, :S/2] = 0

    freq[:, :10] = 0
    freq[:, -10:] = 0
    freq[:10, :] = 0
    freq[-10:, :] = 0


    rects = get_rectangles(freq)

    return rects








def load_positions():
    global POSITIONS

    with open('positions.csv', "r") as fin:
        g = csv.reader(fin, delimiter=',')
        for tokens in g:
            #    0    1      2           3        4     5      6    7       8
            # case, sax, fname, bottom tip, top tip, left, right, top, bottom
            #print tokens
            key = "%s\\study\\%s" % (tokens[0], tokens[1])
            pos = ( int(tokens[7]), int(tokens[8]), int(tokens[5]), int(tokens[6]), 0 )
            POSITIONS[key] = pos








def area_of_intersection(t1, b1, l1, r1,  t2, b2, l2, r2):
    return max(0, min(r1, r2) - max(l1, l2)) * max(0, min(b1, b2) - max(t1, t2))

def area(t, b, l, r):
    return (b - t) * (r - l)




def F1(p, r):
    return 2. * (p * r) / float(p + r)




def prepare_data():
    global TRAIN_SET, TEST_SET

    keys = BUFFERS.keys()
    N = len(keys)

    for i in range(N):
        data = BUFFERS[keys[i]]
        N += len(data[0])


    indices = range(N)

    np.random.shuffle(indices)

    to_train = int(N * .7)
    train_indices = indices[:to_train]
    test_indices  = indices[to_train:]


    # 5 rect features + 4 img features
    columns = 9 + 1

    TRAIN_SET = np.zeros((to_train, columns), dtype=np.float64)
    TEST_SET  = np.zeros((N - to_train, columns), dtype=np.float64)

    idx = 0
    for i in train_indices:
        key = keys[i]

        pos = POSITIONS[key]
        data = BUFFERS[key]

        rects = data[0]
        img_data = data[1]

        for r in rects:
            SP = area(pos[0], pos[1], pos[2], pos[3])
            SR = area(r[0], r[1], r[2], r[3])
            SI = area_of_intersection(pos[0], pos[1], pos[2], pos[3],  r[0], r[1], r[2], r[3])

            P = F1(float(SI) / SP, float(SI) / SR)


            TRAIN_SET[idx,:4] = img_data
            TRAIN_SET[idx,4:-1] = r
            TRAIN_SET[idx,-1] = P
            idx += 1

        
    




#def teach_LR():
    





def main():
    np.random.seed()

    load_positions()

    for k in POSITIONS:
        print k, POSITIONS[k]

    for d in [d for d in os.listdir(PATH_BASE) if not d.startswith(".")]:
        print "Caching %s" % d
        cache(d)


    prepare_data()


if __name__ == "__main__":
    main()
