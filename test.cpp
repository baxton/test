
import sys
import numpy as np


TAG_LEN = 4
TAG_TYPE = np.uint32
TAG_PART_TYPE = np.uint16


def tag(buffer):
    return np.frombuffer(buffer, dtype=TAG_TYPE)[0]

def tag_tostring(tag_id):
    parts = np.frombuffer(np.array([tag_id], dtype=np.uint32), dtype=TAG_PART_TYPE)
    s = "(%04X,%04X)" % tuple(parts)
    return s

def to_word(buffer):
    return np.frombuffer(buffer, dtype=np.uint16)[0]

def to_dword(buffer):
    return np.frombuffer(buffer, dtype=np.uint32)[0]



def buffer_tostring(buffer):
    addr = 0
    count = 0

    msg = "%s type %s\n" % (buffer.shape, buffer.dtype)

    for b in buffer:
        if 0 == (count % 16):
            if count > 0:
                msg += "\n"
 
            msg += "%08X:" % addr
            addr += 16

        if 0 == (count % 2):
            msg += " "

        msg += "%02X" % b
	count += 1

    return msg





def default_tag_handler(file_buffer, cur_idx, dicom_obj, tag_id):
    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    # tag VM - length, 2 bytes
    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += 2

    res += "LEN %02X (%d)" % (length, length)

    if VR == "CS":
        value = np.frombuffer(file_buffer[cur_idx : cur_idx + length], dtype="|S%d" % length)
        res += " [%s...]" % (value[0][:10])

    elif VR == "DS" or \
         VR == "TM":
        value = np.frombuffer(file_buffer[cur_idx : cur_idx + length], dtype="|S%d" % length)
        res += " [%s]" % value[0]

    elif VR == "SH" or \
         VR == "ST" or \
         VR == "LO":
        if length:
            value = np.frombuffer(file_buffer[cur_idx : cur_idx + length], dtype="|S%d" % length)
            res += " [%s]" % value[0]



    cur_idx += length

    print res
   
    return cur_idx 
    



def columns(file_buffer, cur_idx, dicom_obj):
    # 0028,0011
    tag_id = 0x00110028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += 2
    val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.Columns = val
    res += "columns %d" % val

    print res
    
    return cur_idx





def rows(file_buffer, cur_idx, dicom_obj):
    # 0028,0010
    tag_id = 0x00100028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += 2
    val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.Rows = val
    res += "rows %d" % val

    print res
    
    return cur_idx





def number_of_frames(file_buffer, cur_idx, dicom_obj):
    # 0028,0008
    tag_id = 0x00080028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += 2
    val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.NumberOfFrames = val
    res += "number of frames %d" % val

    print res
    
    return cur_idx





def planar_configuration(file_buffer, cur_idx, dicom_obj):
    # 0028,0006
    tag_id = 0x00060028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += 2
    val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.PlanarConfiguration = val
    res += "planar configuration %d" % val

    print res
    
    return cur_idx





def photometric_interpretation(file_buffer, cur_idx, dicom_obj):
    # 0028,0004
    tag_id = 0x00040028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    length = 2
    cur_idx += 2
    val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.PhotometricInterpretation = val
    res += "photometric interpretation %d" % val

    print res
    
    return cur_idx





def samples_per_pixel(file_buffer, cur_idx, dicom_obj):
    # 0028,0002
    tag_id = 0x00020028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = 2
    cur_idx += 2
    val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.SamplesPerPixel = val
    res += "samples per pixel %d" % val

    print res
    
    return cur_idx





def lagest_image_pixel_value(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x01060028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = 2
    cur_idx += 2
    lagest_img_pixel_val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.LagestImagePixelValue = lagest_img_pixel_val
    res += "lagest image pixel value %d" % lagest_img_pixel_val

    print res
    
    return cur_idx




def smallest_image_pixel_value(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x01060028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = 2
    cur_idx += 2
    smallest_img_pixel_val = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.SmallestImagePixelValue = smallest_img_pixel_val
    res += "smallest image pixel value %d" % smallest_img_pixel_val

    print res
    
    return cur_idx




def high_bit(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x01020028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = 2
    cur_idx += 2
    high_bit = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.HighBit = high_bit
    res += "high bit %d" % high_bit

    print res
    
    return cur_idx




def pixel_representation(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x01030028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = 2
    cur_idx += 2
    pixel_representation = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.PixelRepresentation = pixel_representation
    res += "pixel representation %d" % pixel_representation

    print res
    
    return cur_idx





def bits_per_pixel(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x01010028

    res = tag_tostring(tag_id) + " "

    # tag VR - data type, 2 bytes
    VR = chr(file_buffer[cur_idx + 0]) + chr(file_buffer[cur_idx + 1])

    res += VR + " "
    cur_idx += 2

    length = 2
    cur_idx += 2
    bits_stored = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += length

    dicom_obj.BitsStored = bits_stored
    res += "bits stored %d" % bits_stored

    print res

    return cur_idx




def pixel_data_uint16(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x00107FE0

    




def image_type(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x00080008

    res = tag_tostring(tag_id) + " (Image Type) CS "

    cur_idx += 2    # skip VR

    length = to_word(file_buffer[cur_idx : cur_idx + 2])
    cur_idx += 2

    res += "LEN %02X (%d)" % (length, length)
    cur_idx += length

    print res

    return cur_idx


def meta_info_version(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x00010002

    res = tag_tostring(tag_id) + " (meta info ver) OB "

    cur_idx += 2    # skip VR

    length = 8

    res += "LEN %02X (%d)" % (length, length)
    cur_idx += length

    print res

    return cur_idx



def reference_image_sequence(file_buffer, cur_idx, dicom_obj):
    tag_id = 0x11400008

    res = tag_tostring(tag_id) + " (img seq) SQ "
    cur_idx += 8

    print res

    return cur_idx




def SQ_data_item(file_buffer, cur_idx, dicom_obj):
    tag_id = 0xE000FFFE

    res = tag_tostring(tag_id) + " (SQ data item begin) "
    length = to_dword(file_buffer[cur_idx : cur_idx + 4])
    cur_idx += 4

    res += "LEN %02X (%d)" % (length, length)

    print res

    return cur_idx





DicomDispatcher = {

    # 0008,0008 Image Type
    0x00080008 : image_type,

    # 0008,1140 Referenced Image Sequence
    0x11400008 : reference_image_sequence,

    # 0002,0001 Meta Information Version
    0x00010002 : meta_info_version, 


    # SQ
    0xE000FFFE : SQ_data_item,

    # Pixels

    0x00020028 : samples_per_pixel,
    0x00040028 : photometric_interpretation,
    0x00060028 : planar_configuration,
    0x00080028 : number_of_frames,
    0x00100028 : rows,
    0x00110028 : columns,

    0x01010028 : bits_per_pixel,
    0x01020028 : high_bit,
    0x01030028 : pixel_representation,
    0x01060028 : smallest_image_pixel_value,
    0x01070028 : lagest_image_pixel_value,
    
    # 7FE0,0010 Pixel Data
    0x00107FE0 : pixel_data_uint16,

}






class DICOM(object):
    def __init__(self):
        pass

    
# end of DICOM




def process_tags(file_buffer, cur_idx, dicom_obj):
    total_size = file_buffer.shape[0]

    while cur_idx < total_size:
        tag_id = tag(file_buffer[cur_idx : cur_idx + TAG_LEN])
        cur_idx += TAG_LEN

        if tag_id in DicomDispatcher:
            cur_idx = DicomDispatcher[tag_id](file_buffer, cur_idx, dicom_obj)

        else:
            # tag doesn't have a specialized handler
            # do default processing
            cur_idx = default_tag_handler(file_buffer, cur_idx, dicom_obj, tag_id)





def process(file_buffer):
    dicom_obj = DICOM

    cur_idx = 0

    # 
    # skip Preamble (128 bytes)
    #
    cur_idx += 128

    #
    # read DICM
    #
    if 'D' != chr(file_buffer[cur_idx + 0]) and \
       'I' != chr(file_buffer[cur_idx + 1]) and \
       'C' != chr(file_buffer[cur_idx + 2]) and \
       'M' != chr(file_buffer[cur_idx + 3]):
        raise Exception("not DICOM format")
    
    cur_idx += 4
    print "DICM"

    #
    # repetitivly process tags
    #
    process_tags(file_buffer, cur_idx, dicom_obj)


    return dicom_obj







def main():
    fname = sys.argv[1]
    file_buffer = np.fromfile(fname, dtype=np.uint8, sep='')

    print buffer_tostring(file_buffer[:150])

    dicom_obj = process(file_buffer)




#
# run
#
main()
