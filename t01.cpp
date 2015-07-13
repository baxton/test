

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>


using namespace std;


const unsigned int mask = 0x100812;



string replaceBLUE(string& str) {

    const char* p = str.c_str();
    const char* beg = p;

    string result;
    unsigned int m = 0x00;

    while (*p) {
        if (' ' == *p) {
            if (m != mask)
                result.append(beg, p+1);
            else
                result.append("XXXX ");

            m = 0x00;
            beg = p + 1;
        }
        else {
            m |= 1 << (int)(*p - 'A');
        }
        ++p;
    }

    if (m) {
        if (m == mask)
            result.append("XXXX");
        else
            result.append(beg, p);
    }


    return result;
}




int main() {

    string in_str;
    getline(cin, in_str);


    cout << replaceBLUE(in_str) << endl;

    return 0;
}
