
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


typedef double DATATYPE;

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


    void* ann_create(int* vec, int size) {
        ma::random::seed();
        std::vector<int> sizes;
        for (int i = 0; i < size; ++i)
            sizes.push_back(vec[i]);

        ma::ann_leaner<DATATYPE>* ann = new ma::ann_leaner<DATATYPE>(sizes);
        return ann;
    }


    void ann_fit(void* ann, const DATATYPE* X, const DATATYPE* Y, int rows, DATATYPE* alpha, DATATYPE lambda, int epoches, DATATYPE* cost) {

        int cost_cnt = 0;
        DATATYPE prev_cost = 999.;

        vector<DATATYPE> ww, bb;

        for (int e = 0; e < epoches; ++e) {
//            static_cast< ma::ann_leaner<DATATYPE>* >(ann)->save_weights(ww, bb);
            *cost = static_cast< ma::ann_leaner<DATATYPE>* >(ann)->fit_minibatch(X, Y, rows, *alpha, lambda);
/*
            if (prev_cost < *cost) {
                DATATYPE d = *alpha / 100.;
                *alpha -= d;
                static_cast< ma::ann_leaner<DATATYPE>* >(ann)->restore_weights(ww, bb);
            }
            else {
                prev_cost = *cost;
            }
*/
        }

    }


    void ann_free(void* ann) {
        delete static_cast< ma::ann_leaner<DATATYPE>* >(ann);
    }

    void ann_predict(void* ann, const DATATYPE* X, DATATYPE* predictions, int rows) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->predict(X, predictions, rows);
    }



}




