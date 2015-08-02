
//
//  g++ -DTESTS --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 proto.cpp -o proto.exe
//  g++ -DEMUL --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 proto.cpp -o proto.exe
//




#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iterator>
#include <memory>

#include <fstream>
#include <iostream>
#include <string>




typedef double value_t;


namespace rnd {

    void seed() { ::srand(time(NULL)); }

    value_t rand() {
        return (value_t)::rand() / RAND_MAX;
    }


}


namespace linalg {

value_t dot(const value_t* v1, const value_t* v2, int size) {
    value_t r = 0.;
    for (int i = 0; i < size; ++i) {
        r += v1[i] * v2[i];
    }
    return r;
}

void mul_scalar(value_t scalar, const value_t* v, value_t* r, int size) {
    for (int i = 0; i < size; ++i) {
        r[i] = v[i] * scalar;
    }
}


}


namespace stat {


template<typename T>
void get_stat_online(const T* vec, size_t size, value_t& m, value_t& v, value_t& s, value_t& k) {
    value_t n, M1, M2, M3, M4;
    value_t delta, delta_n, delta_n2, term1;

    // init
    n = M1 = M2 = M3 = M4 = 0.;

    // process
    for (size_t i = 0; i < size; ++i) { 
        value_t n1 = n++;
        delta = value_t(vec[i]) - M1;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        M1 += delta_n;
        M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
        M2 += term1;
    }

    m = M1;
    v = M2 / (n - 1.0);
    s = ::sqrt(n) * M3 / ::pow(M2, 1.5);
    k = n * M4 / (M2 * M2) - 3.0;
}

}


namespace optimize {


value_t reg_h(const value_t* theta, const value_t* x, int columns) {
    value_t r = linalg::dot(x, theta, columns);
    return r;
}


value_t reg_cost(const value_t* theta, const value_t* x, value_t* grad_x, value_t y, int columns) {
    // calc logistic val
    value_t h = reg_h(theta, x, columns);

    // calc cost part
    value_t delta = h - y;
    value_t cost = delta * delta / 2.;

    // calc gradient part
    linalg::mul_scalar(delta, x, grad_x, columns);

    return cost;
}


void minimize_gc(value_t* theta, const value_t* x, int columns, const value_t y, int max_iterations) {
    value_t grad[columns];

    value_t e = 0.0001;
    value_t a = .5;

    value_t cost = reg_cost(theta, x, grad, y, columns);

    int cur_iter = 0;

    while (cost > e && cur_iter < max_iterations) {
        ++cur_iter;

        for (int i = 0; i < columns; ++i) {
            theta[i] = theta[i] - a * grad[i];
        }

        value_t new_cost = reg_cost(theta, x, grad, y, columns);

        if (cost < new_cost)
            a /= 2.;

        cost = new_cost;
    }
}

inline
value_t predict(const value_t* theta, const value_t* x, int columns) {
    return reg_h(theta, x, columns);
}



}



value_t poli(value_t x1, value_t x2, value_t x3) {
    value_t y = 2. - x1 + 3. * x2 * x2 - 2.5 * x3 * x3 * x3; 
    return y;
}



int main() {

    rnd::seed();

    std::vector<value_t> X;
    std::vector<value_t> Y;

    size_t N = 2000;
    
    for (size_t n = 0; n < N; ++n) {
        X.push_back(1.);
        value_t x = rnd::rand() - .5;
        X.push_back(x);
        X.push_back(x*x);
        X.push_back(x*x*x);

        size_t idx = n * 4;

        value_t y = poli(x,x,x); // / 2000.;
        Y.push_back(y);

        std::cout << X[n*4] << " " << X[n*4+1] << " " << X[n*4+2] << " " << X[n*4+3] << " = " << Y[n] << std::endl;
    }

    std::vector<value_t> theta;
    for (size_t i = 0; i < 4; ++i)
        theta.push_back(rnd::rand() - .5);


    //
    for (size_t e = 0; e < 10; ++e) {
        for (size_t r = 0; r < 1500; ++r) {
            size_t idx = r * 4;
            optimize::minimize_gc(&theta[0], &X[idx], 4, Y[r], 1);
        }
    }

    // 
    size_t count = 0;
    value_t s = 0;
    for (size_t r = 1500; r < N; ++r) {
        size_t idx = r * 4;
        value_t p = optimize::predict(&theta[0], &X[idx], 4);
        s = (p - Y[r]) * (p - Y[r]);
        count += 1;

        std::cout << p << "\t" << Y[r] << std::endl;
    }

    std::cout << theta[0] << " " << theta[1] << " " << theta[2] << " " << theta[3] << std::endl;
    std::cout << "2 (-1) 3 (-2.5)" << std::endl;
    std::cout << "AVR ERR: " << (s / count) << std::endl;
    
}











