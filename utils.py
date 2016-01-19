

import os
import sys
import numpy as np
import ctypes


SEP = os.path.sep


DICOM_DLL = ctypes.cdll.LoadLibrary("dicom.dll")


class DICOM(object):
    def __init__(self):
        # meta data access offsets
        self.MDSeriesDescription       = 0
        self.MDPixelRepresentation     = 1
        self.MDBitsStored              = 2
        self.MDBitsAllocated           = 3
        self.MDColumns                 = 4
        self.MDRows                    = 5
        self.MDGender                  = 6
        self.MDAge                     = 7
        self.MDLargestPixelValue       = 8
        self.MDSliceThickness          = 9
        self.MDFlipAngle               = 10
        self.MDImagePositionPatient    = 11   # - 13
        self.MDImageOrientationPatient = 14   # - 19


    def fromfile(self, fname):
        self.fname = fname

        # read row content
        # print "read file", fname
        self.buffer = np.fromfile(self.fname, dtype=np.uint8, sep='')

        # process in C++ and get object's handler
        self.dicom = ctypes.c_void_p(0)
        DICOM_DLL.dicom_fromfile(ctypes.c_void_p(self.buffer.ctypes.data), 
                                 ctypes.c_int(self.buffer.shape[0]),
                                 ctypes.c_void_p(ctypes.addressof(self.dicom)))
    

    def free(self):
        DICOM_DLL.dicom_free(self.dicom)
        pass



    def Rows(self):
        rows = ctypes.c_int(0)
        DICOM_DLL.dicom_rows(self.dicom,
                             ctypes.c_void_p(ctypes.addressof(rows)))
        return rows.value


    def Columns(self):
        columns = ctypes.c_int(0)
        DICOM_DLL.dicom_columns(self.dicom,
                                ctypes.c_void_p(ctypes.addressof(columns)))
        return columns.value

    def img_buffer(self):
        img_length = ctypes.c_int(0)
        null = ctypes.c_void_p(0)

        DICOM_DLL.dicom_buffer(self.dicom,
                               null,
                               ctypes.c_void_p(ctypes.addressof(img_length)))
        buffer = np.zeros((img_length.value,), dtype=np.uint16)
        DICOM_DLL.dicom_buffer(self.dicom,
                               ctypes.c_void_p(buffer.ctypes.data),
                               null)
        buffer = buffer.reshape((self.Rows(), self.Columns()))
        return buffer


    def img_metadata(self):
        length = ctypes.c_int(0)
        null = ctypes.c_void_p(0)
 
        DICOM_DLL.dicom_metadata(self.dicom,
                                 null,
                                 ctypes.c_void_p(ctypes.addressof(length)))
        buffer = np.zeros((length.value,), dtype=np.float64)
        DICOM_DLL.dicom_metadata(self.dicom,
                                 ctypes.c_void_p(buffer.ctypes.data),
                                 null)
        return buffer



    def verbose(self, val):
        DICOM_DLL.dicom_verbose(ctypes.c_int(1 if val else 0))



        





def main():
    fname = sys.argv[1]
    dicom = DICOM() 
    dicom.fromfile(fname)
    #
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(dicom.img_buffer())
    plt.show()

    metadata = dicom.img_metadata()
    print metadata
    print metadata.shape 

    print "Rows", metadata[dicom.MDRows]
    print "ImageOrientationPatient", metadata[dicom.MDImageOrientationPatient], metadata[dicom.MDImageOrientationPatient+1], metadata[dicom.MDImageOrientationPatient+2], metadata[dicom.MDImageOrientationPatient+3], metadata[dicom.MDImageOrientationPatient+4], metadata[dicom.MDImageOrientationPatient+5]

    #
    dicom.free()




if __name__ == "__main__":
    main()     

