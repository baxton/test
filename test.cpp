

import sys
import numpy as np
import ctypes


DICOM_DLL = ctypes.cdll.LoadLibrary("dicom.dll")


class DICOM(object):
    def __init__(self):
        pass

    def fromfile(self, fname):
        self.fname = fname

        # read row content
        print "read file", fname
        self.buffer = np.fromfile(self.fname, dtype=np.uint8, sep='')
        print self.buffer.tostring()

        # process in C++ and get object's handler
        self.dicom = ctypes.c_void_p(0)
        DICOM_DLL.dicom_fromfile(ctypes.c_void_p(self.buffer.ctypes.data), 
                                 ctypes.c_int(self.buffer.shape[0]),
                                 ctypes.addressof(self.dicom))
    

    def free(self):
        DICOM_DLL.dicom_free(self.dicom)













def main():
    fname = sys.argv[0]
    dicom = DICOM() 
    dicom.fromfile(fname)
    dicom.free()




if __name__ == "__main__":
    main()     
