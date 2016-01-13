

import os
import sys
import numpy as np
import scipy as sp
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dicom


dir = ""
fnames = None
files_num = 0

buffers = {}

def set_cls(buffer, check, r, c, cls, counters):
    if 0 == check[r,c]:
        return False

    if r > 0 and buffer[r-1,c] > 0:
        buffer[r,c] = buffer[r-1,c]
        counters[buffer[r,c]] += 1
        return False
    

    if c > 0 and buffer[r,c-1] > 0:
        buffer[r,c] = buffer[r,c-1]
        counters[buffer[r,c]] += 1
        return False


    buffer[r,c] = cls
    counters[cls] = 1

    return True



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


    for r in range(10, freq.shape[0] - 10):
        if freq[r,:].sum() > 0:
            top_r = r
            break
    
    for r in range(freq.shape[0] - 10, 10, -1):
        if freq[r,:].sum() > 0:
            bottom_r = r
            break
    
    for c in range(10, freq.shape[1] - 10):
        if freq[:,c].sum() > 0:
            left_c = c
            break
    
    for c in range(freq.shape[1] - 10, 10, -1):
        if freq[:,c].sum() > 0:
            right_c = c
            break
    

    print top_r, bottom_r, left_c, right_c


    freq = freq.reshape(shape3D)


    plt.clf()
    ax = plt.subplot(121)
    plt.imshow(buffers[0])
    ax.add_patch(patches.Rectangle((left_c, top_r), (right_c - left_c), (bottom_r - top_r), fill=False, linewidth=2, edgecolor='red'))

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
