

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>


using namespace std;


const unsigned int mask = 0x100812; // mask for all anagrams of BLUE, but also the same for "BBBLUE" etc, 
                                    //as does not count the number of letteres in a word



string replaceBLUE(string& str) {

    string result;

    const char* p = str.c_str();
    const char* beg = p;
    unsigned int m = 0x00;

    while (*p) {
        if (' ' == *p) {
            if (m == mask && 4 == distance(beg, p))
                result.append("XXXX ");
            else
                result.append(beg, p+1);

            m = 0x00;
            beg = p + 1;
        }
        else {
            m |= 1 << (int)(*p - 'A');
        }
        ++p;
    }

    if (m) {
        if (m == mask && 4 == distance(beg, p))
            result.append("XXXX");
        else
            result.append(beg, p);
    }


    return result;
}




int main() {

    string in_str;
    getline(cin, in_str);


    cout  << "[" << replaceBLUE(in_str) << "]" << endl;

    return 0;
}
