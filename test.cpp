import os
import sys
import csv
import numpy as np
import matplotlib
import matplotlib.patches as patches


matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
#import dicom 
from dicom_wr import DICOM




SEP = os.path.sep


FNAME = "positions.csv"
HEADER = "CASE,SAX,FILE,B_TIP,T_TIP,LEFT,RIGHT,TOP,BOTTOM"
DATA = {}
STOP = False
START = None
PROCESSED = set()


dicom = None


file_idx = 0
files_num = 0
dir = ""
fnames = None
fig = None
img_obj = None


buffers = {}




def dump(event):
    global STOP

    print "Dumping..."
    with open(FNAME, "a+") as fout:
        for k, v in DATA.items():
            fout.write("%s\n" % v)

    STOP = True




def onclick(event):
    #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
    global START
    START = (event.xdata, event.ydata)
    print "button pressed %s" % str(START)

    circle = patches.Circle((event.xdata, event.ydata), radius=2, color='r')
    ax = plt.subplot(111)
    ax.add_artist(circle)
    plt.gcf().canvas.draw()


def onrelease(event):
    global DATA

    tokens = CUR_FILE.split(SEP)
    top = min(START[1], event.ydata)
    bottom = max(START[1], event.ydata)
    left = min(START[0], event.xdata)
    right = max(START[0], event.xdata)
    data_str = "%s,%s,%s,%d,%d,%d,%d" % (tokens[-4], tokens[-2], tokens[-1], \
                                             left,right,top,bottom)
    key = "%s_%s" % (tokens[-4], tokens[-2])
    DATA[key] = data_str
    print "button released %s" % str((event.xdata, event.ydata,))

    circle = patches.Circle((event.xdata, event.ydata), radius=2, color='r')
    ax = plt.subplot(111)
    ax.add_artist(circle)
    plt.gcf().canvas.draw()











fig = plt.figure()
ax = fig.add_subplot(111)

def add_frame():
    global file_idx, img_obj, buffers

    img_buffer = buffers[file_idx]

    if img_obj == None:
        img_obj = plt.imshow(img_buffer)
    else:
        img_obj.set_data(img_buffer)

    file_idx += 1
    if file_idx == files_num:
        file_idx = 0

    fig.canvas.draw()
    fig.canvas.manager.window.after(100, add_frame)




def main():
    global dir, fnames, fig, files_num, buffers, dicom, img_obj
    global CUR_FILE, PROCESSED

    PATH_BASE  = ".." + SEP
    PATH_DATA  = PATH_BASE + "data" + SEP + "train" + SEP
    PATH_STUDY = "study" + SEP

    if os.path.exists(FNAME):
        with open(FNAME, "r") as fin:
            g = csv.reader(fin)
            for tokens in g:
                PROCESSED.add(tokens[2])
                #print tokens[2]

    file_names = []

    for d1 in [d for d in os.listdir(PATH_DATA) if not d.startswith(".")]:
        path = PATH_DATA + d1 + SEP + PATH_STUDY
        for d2 in [d for d in os.listdir(path) if not d.startswith(".")]:
            path2 = path + d2 + SEP
            files = [f for f in os.listdir(path2) if not f.startswith(".") and "dcm" in f]
            if len(files):
                #print path2 + files[0]
                if not files[0] in PROCESSED:
                    file_names.append(path2 + files[0])

    sh2D = None
    sh3D = None


    cnt = 0

    for f in file_names:
        CUR_FILE = f
        cnt += 1
        print ":", CUR_FILE, cnt, "out of", len(file_names)

        tokens = f.split(SEP)
        dir = SEP.join(tokens[:-1]) + SEP


        fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
        files_num = len(fnames)

        buffers.clear()
        for i, f in enumerate(fnames):
            if i not in buffers:
                dicom = DICOM()
                dicom.verbose(0)

                dicom.fromfile(dir + f)
                buffers[i] = dicom.img_buffer()

                dicom.free()
                dicom = None

        img_obj = None
        plt.clf()
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        cid1 = fig.canvas.mpl_connect("button_press_event", onclick)
        cid2 = fig.canvas.mpl_connect("button_release_event", onrelease)
        cid3 = fig.canvas.mpl_connect("key_press_event", dump)
        #win = fig.canvas.manager.window
        fig.canvas.manager.window.after(100, add_frame)
        plt.show()
        
        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)
        fig.canvas.mpl_disconnect(cid3)
        fig.clear()
        
        if STOP:
            break

    dump(None)





if __name__ == "__main__":
    main()
