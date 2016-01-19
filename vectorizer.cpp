/*
 *
 * DICOM file parser
 *
 * Author: Maxim Alekseykin
 *
 * Initial version: 20160115
 *
 */

// dicom.cpp

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <iomanip>
#include <string>
#include <vector>

using namespace std;





typedef unsigned int DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;



bool VERBOSE = true;




enum SerDescType {
    NONE = 0,
    SAX = 1,
    CH2 = 2,
    CH4 = 3
};
const char* SerDescTypeStr[] = {"NONE", "SAX", "2CH", "4CH"};
 
    


struct DICOM {
    SerDescType SeriesDescription;

    DWORD PixelRepresentation;
    DWORD BitsStored;
    DWORD BitsAllocated;
    DWORD Columns;
    DWORD Rows;
    DWORD Gender;
    DWORD Age;
    DWORD LargestPixelValue;
    double SliceThickness;
    double FlipAngle;
    

    vector<double> ImagePositionPatient;
    vector<double> ImageOrientationPatient;

    auto_ptr<WORD> img_buffer;
    DWORD img_length;
};





inline
DWORD to_dword(const BYTE* buffer) {
    return *(const DWORD*)buffer;
}


inline
WORD to_word(const BYTE* buffer) {
    return *(const WORD*)buffer;
}



void split(const char* src, size_t len, const char sep, vector<string>& tokens) {
    const char* beg = src;
    size_t size = 0;
    for (size_t i = 0; i < len; ++i) {
        if (sep == src[i]) {
            tokens.push_back(string(beg, size));
            beg = &src[i+1];
            size = 0;
        }
        else if (' ' == src[i]) {
            break;
        }
        else {
            ++size;
        }
    }
    if (size) {
        tokens.push_back(string(beg, size));
    }
}




size_t default_handler(const BYTE* buffer, size_t cur_idx, DICOM& dicom, DWORD tag_id, string* dst) {
    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    if (dst) {
        if (*(WORD*)VR == *(WORD*)"CS" ||
            *(WORD*)VR == *(WORD*)"DS" ||
            *(WORD*)VR == *(WORD*)"TM" ||
            *(WORD*)VR == *(WORD*)"IS" ||
            *(WORD*)VR == *(WORD*)"AS" ||
            *(WORD*)VR == *(WORD*)"AE" ||
            *(WORD*)VR == *(WORD*)"SH" ||
            *(WORD*)VR == *(WORD*)"ST" ||
            *(WORD*)VR == *(WORD*)"UI" ||
            *(WORD*)VR == *(WORD*)"PN" ||
            *(WORD*)VR == *(WORD*)"DA" ||
            *(WORD*)VR == *(WORD*)"LO") {
            *dst = string((const char*)&buffer[cur_idx], length);
        }
    }

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " "
           << "VR " << VR << " "
           << "LEN " << hex << setw(4) << setfill('0') << length << " (" << dec << length << ") ";
          
        if (*(WORD*)VR == *(WORD*)"CS" ||
            *(WORD*)VR == *(WORD*)"DS" ||
            *(WORD*)VR == *(WORD*)"TM" ||
            *(WORD*)VR == *(WORD*)"IS" ||
            *(WORD*)VR == *(WORD*)"AS" ||
            *(WORD*)VR == *(WORD*)"AE" ||
            *(WORD*)VR == *(WORD*)"SH" ||
            *(WORD*)VR == *(WORD*)"ST" ||
            *(WORD*)VR == *(WORD*)"UI" ||
            *(WORD*)VR == *(WORD*)"PN" ||
            *(WORD*)VR == *(WORD*)"DA" ||
            *(WORD*)VR == *(WORD*)"LO") {
            ss << "[";
            ss.write((const char*)&buffer[cur_idx], length);
            ss << "]";
        } 

        cout << ss.str() << endl;
    }

    cur_idx += length;
    return cur_idx;
}



size_t FlipAngle(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x13140018;

    cur_idx += 2;  // VR
    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    string val((const char*)&buffer[cur_idx], length);
    dicom.FlipAngle = strtod(val.c_str(), NULL);
    cur_idx += length;

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " FlipAngle " << dec << dicom.FlipAngle << endl;
    }

    return cur_idx;
}


size_t ImagePositionPatient(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00320020;

    cur_idx += 2;  // VR
    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    vector<string> tokens;
    split((const char*)&buffer[cur_idx], length, '\\', tokens);
    for (int i = 0; i < tokens.size(); ++i) {
        dicom.ImagePositionPatient.push_back(atof(tokens[i].c_str()));
    }

    cur_idx += length;

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " ImagePositionPatient [";
        for (int i = 0; i < tokens.size(); ++i) {
            cout << dec << dicom.ImagePositionPatient[i] << " ";
        }
        cout << "]" << endl;
    }

    return cur_idx;
}


size_t SliceThickness(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00500018;

    cur_idx += 2;  // VR
    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    string val((const char*)&buffer[cur_idx], length);
    dicom.SliceThickness = strtod(val.c_str(), NULL);
    cur_idx += length;

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " SliceThickness " << dec << dicom.SliceThickness << endl;
    }

    return cur_idx;
}


size_t SeriesDescription(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x103e0008;

    cur_idx += 2;  // VR
    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    if ('s' == buffer[cur_idx] && 'a' == buffer[cur_idx+1] && 'x' == buffer[cur_idx+2]) {
        dicom.SeriesDescription = SAX;
    }
    else if ('2' == buffer[cur_idx] && 'c' == buffer[cur_idx+1] && 'h' == buffer[cur_idx+2]) {
        dicom.SeriesDescription = CH2;
    }
    else if ('4' == buffer[cur_idx] && 'c' == buffer[cur_idx+1] && 'h' == buffer[cur_idx+2]) {
        dicom.SeriesDescription = CH4;
    }
    else {
        dicom.SeriesDescription = NONE;
    }

    cur_idx += length;

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " SeriesDescription " << SerDescTypeStr[dicom.SeriesDescription] << endl;
    }

    return cur_idx;
}


size_t ImageOrientationPatient(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00370020;

    cur_idx += 2;  // VR
    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    vector<string> tokens;
    split((const char*)&buffer[cur_idx], length, '\\', tokens);
    for (int i = 0; i < tokens.size(); ++i) {
        dicom.ImageOrientationPatient.push_back(atof(tokens[i].c_str()));
    }

    cur_idx += length;

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " ImageOrientationPatient [";
        for (int i = 0; i < tokens.size(); ++i) {
            cout << dec << dicom.ImageOrientationPatient[i] << " ";
        }
        cout << "]" << endl;
    }

    return cur_idx;
}




    


size_t meta_info_version(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00010002;
    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " meta info" << endl;
    }
    cur_idx += 8 + 2;
    return cur_idx;
}

size_t reference_image_sequence(const BYTE* buffer, size_t cur_idx, DICOM& dicom, DWORD tag_id) {

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " (img seq) SQ" << endl;
    }

    cur_idx += 8;
    return cur_idx;
}



size_t SQ_data_item(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0xE000FFFE;

    DWORD length = to_dword(&buffer[cur_idx]);
    cur_idx += 4;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " SQ data item LEN " << hex << setw(4) << setfill('0') << length << dec << " (" << length << ")";
        cout << ss.str() << endl;
    }

    return cur_idx;
}

size_t pixel_representation(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x01030028;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    size_t length = 2;
    cur_idx += 2;
    WORD val = to_word(&buffer[cur_idx]);
    cur_idx += length;

    dicom.PixelRepresentation = val;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " pixel representation " << dec << val;
        cout << ss.str() << endl;
    }

    return cur_idx;
}

size_t  bits_per_pixel(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x01010028;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    size_t length = 2;
    cur_idx += 2;
    WORD val = to_word(&buffer[cur_idx]);
    cur_idx += length;

    dicom.BitsStored = val;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " bits stored " << dec << val;
        cout << ss.str() << endl;
    }

    return cur_idx;
}

size_t largest_image_pixel_value(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x01070028;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    size_t length = 2;
    cur_idx += 2;
    WORD val = to_word(&buffer[cur_idx]);
    cur_idx += length;

    dicom.LargestPixelValue = val;

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id << " " << VR
             << " largest pixel value " << dec << val
             << endl;
    }

    return cur_idx;
}



size_t bits_allocated(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x01000028;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    size_t length = 2;
    cur_idx += 2;
    WORD val = to_word(&buffer[cur_idx]);
    cur_idx += length;

    dicom.BitsAllocated = val;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " bits allocated " << dec << val;
        cout << ss.str() << endl;
    }

    return cur_idx;
}

size_t columns(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00110028;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    size_t length = 2;
    cur_idx += 2;
    WORD val = to_word(&buffer[cur_idx]);
    cur_idx += length;

    dicom.Columns = val;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " columns " << dec << val;
        cout << ss.str() << endl;
    }

    return cur_idx;
}

size_t rows(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00100028;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++]; 

    size_t length = 2;
    cur_idx += 2;
    WORD val = to_word(&buffer[cur_idx]);
    cur_idx += length;
    
    dicom.Rows = val;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " rows " << dec << val;
        cout << ss.str() << endl;
    }

    return cur_idx;
}


size_t gender(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00400010;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++]; 

    size_t length = to_word(&buffer[cur_idx]);
    cur_idx += 2;
    char val = buffer[cur_idx];
    cur_idx += length;
    
    dicom.Gender = val == 'F' ? 1 : -1;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " gender " << val;
        cout << ss.str() << endl;
    }

    return cur_idx;
}

size_t age(const BYTE* buffer, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x10100010;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++]; 

    size_t length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    int num = 0;
    char tenor = 'Y';
    for (int i = 0; i < length; ++i) {
        char c = buffer[cur_idx + i];
        if ('0' <= c && c <= '9') {
            num = num * 10 + (int)(c - '0');
        }
        else {
            tenor = buffer[cur_idx + i];
            break;
        }
    }   

    if (tenor == 'Y')
        num *= 12;
 
    dicom.Age = num;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " " << VR;
        ss << " age  ";
        ss.write((const char*)&buffer[cur_idx], length);
        ss << " (" << dec << num << " M)";
        cout << ss.str() << endl;
    }

    cur_idx += length;

    return cur_idx;
}



size_t pixel_data_uint16(const BYTE* buffer, size_t total_size, size_t cur_idx, DICOM& dicom) {
    DWORD tag_id = 0x00107FE0;

    DWORD bytes_per_pixel = dicom.BitsAllocated / 8;
    DWORD img_len = dicom.Rows * dicom.Columns;
    DWORD length = bytes_per_pixel * img_len;

    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    cur_idx += 6;

    DWORD tmp = total_size - cur_idx;

    dicom.img_buffer = auto_ptr<WORD>(new WORD[img_len]);
    dicom.img_length = img_len;
    memcpy(dicom.img_buffer.get(), &buffer[cur_idx], img_len * sizeof(WORD));
//    for (int i = 0; i < img_len; ++i) {
//        dicom.img_buffer.get()[i] = to_word(&buffer[cur_idx + i*2]);
//    }

    if (VERBOSE) {
        cout << hex << setw(8) << setfill('0') << tag_id
             << " VR " << VR << " LEN "
             << hex << setw(4) << setfill('0') << length << " (" << dec << length << ")"
             << " left in array " << hex << setw(4) << setfill('0') << tmp << " (" << dec << tmp << ")" << endl;
    }

    cur_idx += length;
    return cur_idx;
}



void process_tags(const BYTE* buffer, size_t total_size, size_t cur_idx, DICOM& dicom) {
    while (cur_idx < total_size) {
        DWORD tag_id = to_dword(&buffer[cur_idx]);
        cur_idx += 4;

        switch (tag_id) {
        case 0x13140018:
            cur_idx = FlipAngle(buffer, cur_idx, dicom);
            break;

        case 0x00500018:
            cur_idx = SliceThickness(buffer, cur_idx, dicom);
            break;

        case 0x103e0008:
            cur_idx = SeriesDescription(buffer, cur_idx, dicom);
            break;

        case 0x01070028:
            cur_idx = largest_image_pixel_value(buffer, cur_idx, dicom);
            break;

        case 0x00107FE0:
            cur_idx = pixel_data_uint16(buffer, total_size, cur_idx, dicom);
            break;

        case 0x00400010:
            cur_idx = gender(buffer, cur_idx, dicom);
            break;

        case 0x10100010:
            cur_idx = age(buffer, cur_idx, dicom);
            break;

        case 0x00100028:
            cur_idx = rows(buffer, cur_idx, dicom);
            break;

        case 0x00110028:
            cur_idx = columns(buffer, cur_idx, dicom);
            break;

        case 0x01000028:
            cur_idx = bits_allocated(buffer, cur_idx, dicom);
            break;

        case 0x01010028:
            cur_idx = bits_per_pixel(buffer, cur_idx, dicom);
            break;

        case 0x01030028:
            cur_idx = pixel_representation(buffer, cur_idx, dicom);
            break;

        case 0xE000FFFE:
            cur_idx = SQ_data_item(buffer, cur_idx, dicom);
            break;

        case 0x11400008:
        case 0x11200008:
        case 0x11100008:
            cur_idx = reference_image_sequence(buffer, cur_idx, dicom, tag_id);
            break;

        case 0x00010002:
            cur_idx = meta_info_version(buffer, cur_idx, dicom);
            break;

        case 0x00320020:
            cur_idx = ImagePositionPatient(buffer, cur_idx, dicom);
            break;

        case 0x00370020:
            cur_idx = ImageOrientationPatient(buffer, cur_idx, dicom);
            break;

        default:
            {
                //string val;
                cur_idx = default_handler(buffer, cur_idx, dicom, tag_id, NULL /*&val*/);
            }
        }
    }
}


DICOM* process(const BYTE* buffer, size_t total_size) {
    auto_ptr<DICOM> pdicom(new DICOM());

    size_t cur_idx = 128;

    //
    // read "DICM"
    //
    if ('D' != buffer[cur_idx+0] &&
        'I' != buffer[cur_idx+1] &&
        'C' != buffer[cur_idx+2] &&
        'M' != buffer[cur_idx+3]) {
        cerr << "not DICOM format" << endl;
        return NULL;
    }
    cur_idx += 4;

    if (VERBOSE) {
        cout << "DICM" << endl;
    }

    //
    // read tags
    //
    process_tags(buffer, total_size, cur_idx, *pdicom);

    return pdicom.release();
}














//
// interface to Python
// 
extern "C" {

    void dicom_fromfile(const BYTE* buffer, int total_size, void** pdicom) {
        *pdicom = process(buffer, total_size);
    }

    void dicom_free(void* pdicom) {
        delete static_cast<DICOM*>(pdicom);
    }

    

    void dicom_rows(void* pdicom, int* rows) {
        *rows = static_cast<DICOM*>(pdicom)->Rows;
    }

    void dicom_columns(void* pdicom, int* columns) {
        *columns = static_cast<DICOM*>(pdicom)->Columns;
    }

    void dicom_buffer(void* pdicom, short* buffer, int* size) {
        if (!buffer) {
            *size = static_cast<DICOM*>(pdicom)->img_length;
        }
        else {
            memcpy(buffer, static_cast<DICOM*>(pdicom)->img_buffer.get(), static_cast<DICOM*>(pdicom)->img_length * sizeof(WORD));
        }
    }

    void dicom_metadata(void* pdicom, double* buffer, int* size) {
        if (!buffer) {
            *size = 11 + static_cast<DICOM*>(pdicom)->ImagePositionPatient.size() + static_cast<DICOM*>(pdicom)->ImageOrientationPatient.size();
        }
        else {
            buffer[0] = static_cast<DICOM*>(pdicom)->SeriesDescription;
            buffer[1] = static_cast<DICOM*>(pdicom)->PixelRepresentation;
            buffer[2] = static_cast<DICOM*>(pdicom)->BitsStored;
            buffer[3] = static_cast<DICOM*>(pdicom)->BitsAllocated;
            buffer[4] = static_cast<DICOM*>(pdicom)->Columns;
            buffer[5] = static_cast<DICOM*>(pdicom)->Rows;
            buffer[6] = static_cast<DICOM*>(pdicom)->Gender;
            buffer[7] = static_cast<DICOM*>(pdicom)->Age;
            buffer[8] = static_cast<DICOM*>(pdicom)->LargestPixelValue;
            buffer[9] = static_cast<DICOM*>(pdicom)->SliceThickness;
            buffer[10] = static_cast<DICOM*>(pdicom)->FlipAngle;

            int cnt = 11;
            for (int i = 0; i < static_cast<DICOM*>(pdicom)->ImagePositionPatient.size(); ++i)
                buffer[cnt++] = static_cast<DICOM*>(pdicom)->ImagePositionPatient[i];

            for (int i = 0; i < static_cast<DICOM*>(pdicom)->ImageOrientationPatient.size(); ++i)
                buffer[cnt++] = static_cast<DICOM*>(pdicom)->ImageOrientationPatient[i];

        }
    }
 

    void dicom_verbose(int val) {
        VERBOSE = val;
    }

}













