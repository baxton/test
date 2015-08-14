

#if !defined RANDOM_DOT_HPP
#define RANDOM_DOT_HPP


#include <cstdlib>
#include <ctime>




namespace ma {

namespace random {

    void seed(int s=-1) {
        if (s == -1)
            ::srand(time(NULL));
        else
            ::srand(s);
    }

    int randint() {
#if 0x7FFF < RAND_MAX
        return ::rand();
#else
        return (int)::rand() * ((int)RAND_MAX + 1) + (int)::rand();
#endif
    }


    int randint(int low, int high) {
        int r = randint();
        r = r % (high - low) + low;
        return r;
    }

    void randint(int low, int high, int* numbers, int size) {
        for (int i = 0; i < size; ++i) {
            numbers[i] = randint() % (high - low) + low;
        }
    }

    /*
     * Retuns:
     * k indices out of [0-n) range
     * with no repetitions
     */
    void get_k_of_n(int k, int n, int* numbers) {
        for (int i = 0; i < k; ++i) {
            numbers[i] = i;
        }

        for (int i = k; i < n; ++i) {
            int r = randint(0, i);
            if (r < k) {
                numbers[r] = i;
            }
        }
    }

    template<class T>
    void rand(T* buffer, int size) {
        for (int i = 0; i < size; ++i) {
            T tmp = ::rand();
            buffer[i] = tmp / RAND_MAX;
        }
    }
}

}


#endif
