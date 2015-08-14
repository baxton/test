
//
// g++ -O3 -I. -msse3 ann_classifier.cpp -shared -o ann.dll
// g++ -O3 -I. ann_classifier.cpp -shared -o ann.dll
//

/*
 * Wrapper for Python
 *
 *
 */

#include <cstdio>
#include <vector>
#include <ann.hpp>


typedef float DATATYPE;

extern "C" {

/*
    void* ann_fromfile(const char* fname) {
        FILE* fin = fopen(fname, "rb");
        if (!fin)
            return NULL;

        fseek(fin, 0, SEEK_END);
        size_t size = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        size_t buffer_size = size / sizeof(DATATYPE);
        ma::memory::ptr_vec<DATATYPE> buffer(new DATATYPE[buffer_size]);
        size_t read = fread(buffer.get(), size, 1, fin);

        ma::ann<DATATYPE>* ann = new ma::ann<DATATYPE>(buffer.get());

        return ann;
    }
*/


    void* ann_create() {
        ma::random::seed();

        std::vector<int> sizes;
        sizes.push_back(2);
        sizes.push_back(128);
        sizes.push_back(128);
//        sizes.push_back(50);
//        sizes.push_back(10);
//        sizes.push_back(10);
//        sizes.push_back(10);
//        sizes.push_back(128);
//        sizes.push_back(128);
//        sizes.push_back(128);
        sizes.push_back(3);

        ma::ann_leaner<DATATYPE>* ann = new ma::ann_leaner<DATATYPE>(sizes);
        return ann;
    }


    void ann_fit(void* ann, const DATATYPE* X, const DATATYPE* Y, int rows, DATATYPE* alpha, DATATYPE lambda, int epoches) {

        int cost_cnt = 0;
        DATATYPE prev_cost = 999.;
        DATATYPE cost = 0;

        for (int e = 0; e < epoches; ++e) {
            cost = static_cast< ma::ann_leaner<DATATYPE>* >(ann)->fit_minibatch(X, Y, rows, *alpha, lambda);

            if (prev_cost < cost) {
                if (*alpha > 0.01)
                    *alpha /= 2.;
            }

            prev_cost = cost;

/*
            same_cost_cnt += (0.00001 >= ::fabs(prev_cost - cost));
            if (10 == same_cost_cnt) {
                *alpha = 7.;
                same_cost_cnt = 0;
            }
*/
            if (0 < e && 0 == (e % 100))
                cout << setprecision(16) << cost << " [" << *alpha << "]" << endl;
            if (*alpha == 0.) {
                break;
            }
        }
        cout << setprecision(16) << cost << " [" << *alpha << "]" << endl;

    }


    void ann_free(void* ann) {
        delete static_cast< ma::ann_leaner<DATATYPE>* >(ann);
    }

    void ann_predict(void* ann, const DATATYPE* X, DATATYPE* predictions, int rows) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->predict(X, predictions, rows);
    }



}



