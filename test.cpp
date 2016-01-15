/*
 *
 * DICOM file parser
 *
 * Author: Maxim Alekseykin
 *
 * Initial version: 20160115
 *
 */



#include <cstring>
#include <iostream>
#include <sstream>
#include <memory>
#include <iomanip>

using namespace std;





typedef unsigned int DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;



bool VERBOSE = true;




struct DICOM {

};





inline
DWORD to_dword(const BYTE* buffer) {
    return *(const DWORD*)buffer;
}


inline
WORD to_word(const BYTE* buffer) {
    return *(const WORD*)buffer;
}




size_t default_handler(const BYTE* buffer, size_t cur_idx, DICOM& dicom, DWORD tag_id) {
    char VR[3] = {0,0,0};
    VR[0] = buffer[cur_idx++];
    VR[1] = buffer[cur_idx++];

    WORD length = to_word(&buffer[cur_idx]);
    cur_idx += 2;

    if (VERBOSE) {
        stringstream ss;
        ss << hex << setw(8) << setfill('0') << tag_id << " "
           << "VR " << VR << " "
           << "LEN " << hex << setw(4) << setfill('0') << length << " (" << dec << length << ") ";
          
        if (0 == strcmp(VR, "CS") ||
            0 == strcmp(VR, "DS") ||
            0 == strcmp(VR, "TM") ||
            0 == strcmp(VR, "IS") ||
            0 == strcmp(VR, "AS") ||
            0 == strcmp(VR, "AE") ||
            0 == strcmp(VR, "SH") ||
            0 == strcmp(VR, "ST") ||
            0 == strcmp(VR, "LO")) {
            ss << "[" << setw(length) << &buffer[cur_idx] << "]";
        } 

        cout << ss.str() << endl;
    }

    cur_idx += length;
    return cur_idx;
}




void process_tags(const BYTE* buffer, size_t total_size, size_t cur_idx, DICOM& dicom) {
    while (cur_idx < total_size) {
        DWORD tag_id = to_dword(&buffer[cur_idx]);
        cur_idx += 4;

        switch (tag_id) {
        default:
            cur_idx = default_handler(buffer, cur_idx, dicom, tag_id);
        }
    }
}


DICOM* process(const BYTE* buffer, size_t total_size) {
    auto_ptr<DICOM> pdicom(new DICOM());

    size_t cur_idx = 128;

    //
    // read "DICM"
    //
    cout << (const char)buffer[cur_idx]  << (const char)buffer[cur_idx + 1] << endl;
    if ('D' != buffer[cur_idx++] &&
        'I' != buffer[cur_idx++] &&
        'C' != buffer[cur_idx++] &&
        'M' != buffer[cur_idx++]) {
        cerr << "not DICOM format" << endl;
        return NULL;
    }

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





}












