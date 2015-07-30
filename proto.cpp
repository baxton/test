
//
//  g++ -DTESTS --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 proto.cpp -o proto.exe
//




#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iterator>

#if defined TESTS
#    include <iostream>
#endif




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

void mul_and_add(value_t scalar, const value_t* v, value_t* r, int size) {
    for (int i = 0; i < size; ++i) {
        r[i] += v[i] * scalar;
    }
}


}


namespace stat {

void get_stat_online(const value_t* vec, size_t size, value_t& m, value_t& v, value_t& s, value_t& k) {
    value_t n, M1, M2, M3, M4;
    double delta, delta_n, delta_n2, term1;

    // init
    n = 0.;
    M1 = M2 = M3 = M4 = 0.;

    for (size_t i = 0; i < size; ++i) { 
        value_t n1 = n;
        n++;
        delta = vec[i] - M1;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        M1 += delta_n;
        M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
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

value_t sigmoid(value_t x) {
    x = x < 20 ? (x > -20 ? x : -20) : 20;
    return 1. / (1. + ::exp(-x));
}

value_t logistic_h(const value_t* theta, const value_t* x, int columns) {
    value_t r = linalg::dot(x, theta, columns);
    return sigmoid(r);
}


// for logistic cos func: sigmoid( h(X) )
// processes 1 sample per call
value_t logistic_cost(const value_t* theta, const value_t* x, value_t* grad_x, value_t y, int columns) {
    // calc logistic val
    value_t h = logistic_h(theta, x, columns);

    value_t p1 = h > 0 ? h : 0.0000000000001;
    value_t p2 = (1. - h) > 0 ? (1. - h) : 0.0000000000001;

    // calc cost part
    value_t cost = -y * ::log(p1) - (1. - y) * ::log(p2);

    // calc gradient part
    value_t delta = h - y;
    linalg::mul_and_add(delta, x, grad_x, columns);

    return cost;
}


void minimize_gc(value_t* theta, const value_t* x, int columns, const value_t y, int max_iterations) {
    value_t grad[columns];

    value_t e = 0.0001;
    value_t a = .4;

    value_t cost = logistic_cost(theta, x, grad, y, columns);

    int cur_iter = 0;

    while (cost > e && cur_iter < max_iterations) {
        ++cur_iter;

        for (int i = 0; i < columns; ++i) {
            theta[i] = theta[i] - a * grad[i];
        }

        value_t new_cost = logistic_cost(theta, x, grad, y, columns);

        if (cost < new_cost)
            a /= 2.;

        cost = new_cost;
    }
}

inline
value_t predict(const value_t* theta, const value_t* x, int columns) {
    return logistic_h(theta, x, columns);
}



}




class QuakePredictor {


    int rate;
    int sites;
    std::vector<value_t> sitesLocations;

    size_t columns;
    std::vector<value_t> theta;


    QuakePredictor(const QuakePredictor&);
    QuakePredictor& operator=(const QuakePredictor&);

public:
    QuakePredictor() :
        rate(0),
        sites(0),
        sitesLocations(),
        columns(0),
        theta()
    {
        rnd::seed();
    }
    ~QuakePredictor() {}


    value_t distance(value_t lat1, value_t lon1, value_t lat2, value_t lon2) {
        value_t earthRadius = 6371.01;

        value_t deltaLon = ::fabs( lon1 - lon2 );
        if (deltaLon > 180)
            deltaLon = 360. - deltaLon;

        return earthRadius * ::atan2( ::sqrt( ::pow( ::cos(lat1) * ::sin(deltaLon), 2 ) + ::pow( ::cos(lat2) * ::sin(lat1) - ::sin(lat2) * ::cos(lat1) * ::cos(deltaLon), 2 ) ), ::sin(lat2) * ::sin(lat1) + ::cos(lat2) * ::cos(lat1) * ::cos(deltaLon) );
    }


    value_t is_event(value_t lat, value_t lon, const std::vector<value_t>& quakes) {
        value_t MAX_DIST = 70.;

        size_t count = 0;
        size_t size = quakes.size() / 5;

        for (size_t i = 0; i < size; ++i) {
            value_t dist = distance(lat, lon, quakes[i * 5], quakes[i * 5 + 1]);
            count += dist <= MAX_DIST;
        }

        return count ? 1. : 0.;
    }


    int init(int sampleRate, int numOfSites, const std::vector<double>& sitesData) {
        rate = sampleRate;
        sites = numOfSites;
        std::copy(sitesData.begin(), sitesData.end(), std::back_inserter(sitesLocations));

        // init classifier
        columns = 1 + 1 + 4;
        for (size_t i = 0; i < columns; ++i)
            theta.push_back(rnd::rand());

        return 0;
    }



    //
    // for debugging
    //

    const std::vector<value_t>& get_theta() { return theta; }
};









// ----------------------------------------------------------
// Tests
// ----------------------------------------------------------


#if defined TESTS

void test() {
    value_t X[][3] = {{1, 1, 5},
                   {1, 2, 4},
                   {1, 1, 3},
                   {1, -1, 35},
                   {1, -3, 1},
                  };
    value_t Y[] = {1, 1, 1, 0, 0};


    value_t theta[] = {1, 3, 0};

    size_t columns = 3;
    size_t rows = 5;


    for (size_t e = 0; e < 10; ++e) {
        for (size_t r = 0; r < rows; ++r) {
            optimize::minimize_gc(theta, X[r], columns, Y[r], 5);
        }
    }


    // print
    for (size_t i = 0; i < columns; ++i) {
        std::cout << theta[i] << " ";
    }
    std::cout << std::endl;
}


void test2() {
    QuakePredictor qp;

    std::cout << "MOS-SIN: " << qp.distance(37.6156, 55.7522, 103.8, 1.3667) << std::endl;

}

void test3() {

    std::vector<value_t> sitesData;

    QuakePredictor qp;
    qp.init(0, 0, sitesData);

    const std::vector<value_t>& theta = qp.get_theta();

    for (size_t i = 0; i < theta.size(); ++i)
        std::cout << theta[i] << " ";
    std::cout << std::endl;
}


void test4() {

    value_t a[] = {1,2,3,4,5,6,7,8,9,0};
    size_t size = sizeof(a) / sizeof(value_t);


    value_t m, v, s, k;

    stat::get_stat_online(a, size, m, v, s, k);

    std::cout << "MVSK: " << m << "\t" << v << "\t" << s << "\t" << k << std::endl;
}


int main() {

    test();
    test2();
    test3();
    test4();


    return 0;
}


#endif
















