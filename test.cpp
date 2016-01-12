

import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dicom













def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)






def main():
    PATH_BASE  = "..\\"
    PATH_DATA  = PATH_BASE + "data\\"
    PATH_STUDY = "study\\"

    file_names = []

    for d1 in os.listdir(PATH_DATA):
        path = PATH_DATA + d1 + "\\" + PATH_STUDY
        for d2 in [d for d in os.listdir(path) if not d.startswith(".")]:
            path2 = path + d2 + "\\"
            files = [f for f in os.listdir(path2) if not f.startswith(".") and "dcm" in f]
            if len(files):
                print path2 + files[0]
                file_names.append(path2 + files[0])
            else:
                print path2

    sh2D = None
    sh3D = None

    for f in file_names:
        buf = dicom.img_fromfile(f)
        sh2D = (buf.shape[0], buf.shape[1])
        sh3D = (buf.shape[0], buf.shape[1], 1)

        buf = buf.reshape(sh3D)

        #
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.concatenate((buf,buf,buf), axis=2))

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        cid = fig.canvas.mpl_connect("button_release_event", onclick)

        plt.show()
 
        fig.canvas.mpl_disconnect(cid)
        
        
    



if __name__ == "__main__":
    main()
