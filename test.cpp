

import os
import sys
import csv
import numpy as np
import scipy as sp
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dicom
import blur


dir = ""
fnames = None
files_num = 0

buffers = {}

POSITIONS = {}



def load_positions():
    global POSITIONS

    with open('positions.csv', "r") as fin:
        g = csv.reader(fin, delimiter=',')
        for tokens in g:
            if 7 == len(tokens):
                #    0    1      2     3      4    5       6
                # case, sax, fname, left, right, top, bottom
                key = "%s\\study\\%s" % (tokens[0], tokens[1])
                pos = ( int(tokens[5]), int(tokens[6]), int(tokens[3]), int(tokens[4]), 0 )
                POSITIONS[key] = pos
            else:
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




def get_stat():
    global dir, fnames, files_num, buffers

    path_base = "..\\data\\"

    dicom.VERBOSE = False

    min_rows = 99999
    min_cols = 99999
    max_rows = 0
    max_cols = 0
    min_frames = 99999
    max_frames = 0

    for d in [d for d in os.listdir(path_base) if not d.startswith(".")]:
        path1 = path_base + d + "\\study\\"
        for d2 in [d for d in os.listdir(path1) if not d.startswith(".")]:
            path_final = path1 + d2 + "\\"
            dir = path_final


            fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
            files_num = len(fnames)

            for i, f in enumerate(fnames):
                file_buffer = np.fromfile(dir + f, dtype=np.uint8, sep='')
                dicom_obj = dicom.process(file_buffer)
               
                rows = dicom_obj.Rows
                cols = dicom_obj.Columns

                if min_rows > rows:
                    min_rows = rows
                if min_cols > cols:
                    min_cols = cols
                if max_rows < rows:
                    max_rows = rows
                if max_cols < cols:
                    max_cols = cols

            frames = len(fnames)
       
            if min_frames > frames:
                min_frames = frames
            if max_frames < frames:
                max_frames = frames

    print "STAT"
    print " min rows", min_rows
    print " max rows", max_rows
    print " min cols", min_cols
    print " max cols", max_cols
    print " mim frames", min_frames
    print " max frames", max_frames













def process_and_save(): 
    global dir, fnames, files_num, buffers

    path_base = "..\\data\\"

    dicom.VERBOSE = False

    for d in [d for d in os.listdir(path_base) if not d.startswith(".")]:
        path1 = path_base + d + "\\study\\"
        for d2 in [d for d in os.listdir(path1) if not d.startswith(".")]:
            path_final = path1 + d2 + "\\"
            dir = path_final
            

            fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
            files_num = len(fnames)

            buffers.clear()


            fig = process(ret=True)

            tokens = dir.split("\\")

            fname = "..\\imgs\\%s_%s_%s.png" % (tokens[-4], tokens[-2], tokens[-1])
            fig.savefig(fname)
   





#def process(ret=False): 
#    global dir, fnames, files_num, buffers
#
#
#    dicom.VERBOSE = False
#
#    fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
#    files_num = len(fnames)
#
#    for i, f in enumerate(fnames):
#        if i not in buffers:
#            img_buffer = dicom.img_fromfile(dir + f)
#            buffers[i] = blur.process(img_buffer)
#
#   
#     
#
#
#    prev = buffers[0]
#    shape2D = prev.shape
#    shape3D = (shape2D[0], shape2D[1], 1)
#
#    tmp = np.zeros(shape2D, dtype=float)
#    freq = np.zeros(shape2D, dtype=float)
#
#    cnt = 0
#    thr = 2
#
#    for i in range(1, files_num):
#        cnt += 1
#        if cnt != thr:
#            continue
#        cnt = 0
#        curr = buffers[i]
#        #
#        tmp = (prev - curr) ** 2
##        mv = np.mean(tmp)
#        tmp[tmp < (mv*4)] = 0
#        freq[tmp != 0] += 1
#
#        #
#        prev = curr
#
#
#    freq /= files_num
#    freq *= 100
#
#
#
#
#    for e in range(3):
#        R = shape2D[0]
#        C = shape2D[1]
#        S = 6
#        K = 12 + e
#        res = np.zeros(shape2D, dtype=float)
#        for r in range(R):
#            if (r + S) >= R:
#                continue
#            for c in range(C):
#                if (c + S) >= C:
#                    continue
#                tmp = freq[r : r + S, c : c + S]
#                m = np.mean(tmp)
#    
#                #if K - 1 <= m and m < K + 1:
#                if K < m:
#                    res[r,c] = 100
#                else:
#                    res[r,c] = 0
#    
#        freq[S/2 :, S/2 :] = res[:-S/2, :-S/2]
#        freq[:S/2, :S/2] = 0
#
#    freq[:, :15] = 0
#    freq[:, -15:] = 0
#    freq[:15, :] = 0
#    freq[-15:, :] = 0
#
#    freq[freq != 0] = 1
#
#
#    vert_hist = freq.sum(axis=1)
#    hors_hist = freq.sum(axis=0)
#
##    vbuf = np.zeros(sh2D)
##    hbuf = np.zeros(sh2D)
#
##    for c in range(sh2D[1]):
##        hbuf[-hors_hist[c]]
#
# 
#
#
#    for r in range(10, freq.shape[0] - 10):
#        if freq[r,:].sum() > 0:
#            top_r = r
#            break
#    
#    for r in range(freq.shape[0] - 10, 10, -1):
#        if freq[r,:].sum() > 0:
#            bottom_r = r
#            break
#    
#    for c in range(10, freq.shape[1] - 10):
#        if freq[:,c].sum() > 0:
#            left_c = c
#            break
#    
#    for c in range(freq.shape[1] - 10, 10, -1):
#        if freq[:,c].sum() > 0:
#            right_c = c
#            break
#    
#
##    print top_r, bottom_r, left_c, right_c
#
#
#    freq = freq.reshape(shape3D)
#
#
#   
#
#
#    plt.clf()
#    ax = plt.subplot(221)
#    plt.imshow(buffers[0])
#
#    for r in rects:
#        ax.add_patch(patches.Rectangle((r[2], r[0]), r[3] - r[2], r[1] - r[0], fill=False, linewidth=2, edgecolor='red'))
#
#
#    plt.subplot(222)
#    plt.imshow(np.concatenate((freq,freq,freq), axis=2))
#
#    ax = plt.subplot(224)
#    ax.bar(range(hors_hist.shape[0]), hors_hist, 1)
#
#    if not ret:
#        plt.show()
#    else:
#        return plt.gcf()
#
#
#



def process(ret=False): 
    global dir, fnames, files_num, buffers


    dicom.VERBOSE = False

    fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
    files_num = len(fnames)

    for i, f in enumerate(fnames):
        if i not in buffers:
            img_buffer = dicom.img_fromfile(dir + f)
            buffers[i] = img_buffer

   
     


    prev = buffers[0]
    shape2D = prev.shape
    shape3D = (shape2D[0], shape2D[1], 1)

    tmp = np.zeros(shape2D, dtype=float)
    freq = np.zeros(shape2D, dtype=float)

    cnt = 0
    thr = 2

    for i in range(1, files_num):
        cnt += 1
        if cnt != thr:
            continue
        cnt = 0
        curr = buffers[i]
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


    freq = freq.reshape(shape3D)

    
    tokens = dir.split("/")
    print tokens
    key = "%s\\study\\%s" % (tokens[-4], tokens[-2])
    pos = POSITIONS[key]
    SP = area(pos[0], pos[1], pos[2], pos[3])
    print "real", pos


    plt.clf()
    ax = plt.subplot(121)
    plt.imshow(buffers[0])
   
    for r in rects:
        SR = area(r[0], r[1], r[2], r[3])
        SI = area_of_intersection(pos[0], pos[1], pos[2], pos[3],  r[0], r[1], r[2], r[3])

        if 0 == SI:
            P = 0
        else:
            P = F1(float(SI) / SP, float(SI) / SR)

        ax.add_patch(patches.Rectangle((r[2], r[0]), r[3] - r[2], r[1] - r[0], fill=False, linewidth=2, edgecolor='red'))

        print "rect", P, str(r)

    ax.add_patch(patches.Rectangle((pos[2], pos[0]), pos[3] - pos[2], pos[1] - pos[0], fill=False, linewidth=2, edgecolor='green'))


    plt.subplot(122)
    plt.imshow(np.concatenate((freq,freq,freq), axis=2))

    if not ret:
        plt.show()
    else:
        return plt.gcf()






def main(): 
    global dir, fnames, files_num, buffers

    dir = sys.argv[1]

    if dir == "ALL":
        process_and_save()
        return 0

    if dir == "STAT":
        get_stat()
        return 0

    load_positions()

    dicom.VERBOSE = False

    fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
    files_num = len(fnames)

    for i, f in enumerate(fnames):
        if i not in buffers:
            img_buffer = dicom.img_fromfile(dir + f)
            buffers[i] = img_buffer

    process() 






if __name__ == "__main__":
    main()

