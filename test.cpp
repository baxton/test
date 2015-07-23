
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstring>

using namespace std;


namespace parser {

unsigned int stoui(const char* p, size_t size) {
    // max unsigned int = 4294967295;
    assert(4 == sizeof(unsigned int));  // if this assert fails I need to check the range of possible values

    unsigned long res = 0;

    switch(size) {
    case 10:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 1000000000 * (*p++ - '0');
    case 9:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 100000000 * (*p++ - '0');
    case 8:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 10000000 * (*p++ - '0');
    case 7:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 1000000 * (*p++ - '0');
    case 6:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 100000 * (*p++ - '0');
    case 5:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 10000 * (*p++ - '0');
    case 4:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 1000 * (*p++ - '0');
    case 3:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 100 * (*p++ - '0');
    case 2:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += 10 * (*p++ - '0');
    case 1:
        if (*p < '0' || '9' < *p)
            throw runtime_error("invalid integer number format");
        res += (*p - '0');
    }

    if (res > 4294967295u)
        throw runtime_error("invalid uint value is out of range");

    return (unsigned int)res;    
}


bool parse_message(const char* p, size_t size, char& type, int& id, char& side, int& quantity, int& price) {
    // A,1,A,1,1
    // A,23,S,55,1025
    // max positive int: 2147483647

    if (!p || size < 9)
        return false;

    const int expected_commas = 4;

    const char* end = p + size;

    int comma_num = 0;
    const char* commas[4];

    char tmp_type;
    int tmp_id;
    char tmp_side;
    int tmp_quantity;
    int tmp_price;

    for (size_t i = 0; i < size; ++i) {
        if (',' == p[i]) {
            commas[comma_num++] = &p[i];
        }
    }

    if (comma_num != expected_commas)
        throw runtime_error("invalid message format, there should be 4 commas only");


    if ('A' != p[0] && 'M' != p[0] && 'X' != p[0])
        throw runtime_error("invalid message format");

    tmp_type = *p;

    p = commas[0] + 1;
    size_t tmp_size = commas[1] - p;

    tmp_id = (int)stoui(p, tmp_size);

    p = commas[1] + 1;
    if ('S' != *p && 'B' != *p)
         runtime_error("invalid message side");
    tmp_side = *p;

    p = commas[2] + 1;
    tmp_size = commas[3] - p;
    tmp_quantity = stoui(p, tmp_size);

    p = commas[3] + 1;
    tmp_size = end - p;
    tmp_price = stoui(p, tmp_size);


    type = tmp_type;
    id = tmp_id;
    side = tmp_side;
    quantity = tmp_quantity;
    price = tmp_price;   

    return false; 
}


} // namespace parser



void test_int(const char* p, bool negative) {
    bool result = true;
    size_t size = strlen(p);
    try {
        result = parser::stoui(p, size);
    }
    catch(const exception& e) {
        result = false;
    }

    if (result && !negative || !result && negative) 
        cout << "OK" << endl;
    else
        cout << "ERR" << endl;
}



void test_stoui() {
    // test parsing int val
    for (int i = 0; i < 1000000; ++i)
        parser::stoui("1234567890", 10);
    cout << parser::stoui("1234567890", 10) << endl;

    test_int("-1234567890", true);
    test_int("91234567890", true);
    test_int("123", false);
    test_int("123.45", true);
    test_int("maxim", true);
}

int main() {

    //test_stoui();

    // test parsing messages
    char type, side;
    int id, quantity, price;
    const char* msg = "A,23,S,15,1025";
    size_t size = strlen(msg);
    parser::parse_message(msg, size, type, id, side, quantity, price);
    cout << type << "," << id << "," << side << "," << quantity << "," << price << endl;
    for (int i = 0; i < 1000000; ++i) {
        parser::parse_message(msg, size, type, id, side, quantity, price);
        cout << type << "," << id << "," << side << "," << quantity << "," << price << endl;
    }


    return 0;
}
