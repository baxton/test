
//
//  g++ -DTESTS --std=c++0x -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3 proto.cpp -o proto.exe
//




#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iterator>
#include <memory>

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


    size_t rate;
    size_t sites;
    std::vector<value_t> sitesLocations;

    size_t columns;
    std::vector<value_t> theta;


    size_t features_size;
    std::auto_ptr<value_t> data;


    QuakePredictor(const QuakePredictor&);
    QuakePredictor& operator=(const QuakePredictor&);

public:
    QuakePredictor() :
        rate(0),
        sites(0),
        sitesLocations(),
        columns(0),
        theta(),
        features_size(0),
        data(NULL)
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


        // allocate memory for data
        // Format:
        // [hour #][site #][Y + <14 features>]
        // 14 features:
        //    <always 1><hour><mvsk * 3>
        features_size = 14;
        size_t total_buffer_size = 2160 * sites * (1 + features_size) * sizeof(value_t);
        data = std::auto_ptr<value_t>(new value_t[total_buffer_size]);


        // init classifier
        for (size_t i = 0; i < features_size; ++i)
            theta.push_back(rnd::rand());

        return 0;
    }

    void add_observation(size_t hour, size_t site, const int* raw_data, size_t size, const std::vector<value_t>& quakes) {
        std::vector<value_t> vec(&raw_data[0], &raw_data[size]);

        value_t features[1 + features_size];

        features[0] = is_event(sitesLocations[site * 2], sitesLocations[site * 2 + 1], quakes);

        features[1] = 1.;   // always 1.

        size_t beg = 0;
        size_t vec_size = size / 3;

        stat::get_stat_online(&vec[beg], vec_size, features[3], features[4], features[5], features[6]);
        beg += vec_size;
        stat::get_stat_online(&vec[beg], vec_size, features[7], features[8], features[9], features[10]);
        beg += vec_size;
        stat::get_stat_online(&vec[beg], vec_size, features[11], features[12], features[13], features[14]);


        size_t s_size = (1 + features_size) * sizeof(value_t);
        size_t h_size = sites * s_size;

        std::copy(&features[0], &features[1 + features_size], &data.get()[hour * h_size + site * s_size]);
    }



    void fit_classifier(size_t hour) {
        size_t s_size = (1 + features_size) * sizeof(value_t);
        size_t h_size = sites * s_size;


        for (size_t h = 0; h <= hour; ++h) {
            for (size_t s = 0; s < sites; ++s) {
                value_t y = data.get()[hour * h_size + s * s_size];   // TODO move out of the loop
                value_t* x = &data.get()[h * h_size + s * s_size + 1];

                x[1] = hour - h; 

                optimize::minimize_gc(&theta[0], x, features_size, y, 5);
            }
        }
    }



    void predict(size_t hour, std::vector<double>& predictions) {
        size_t s_size = (1 + features_size) * sizeof(value_t);
        size_t h_size = sites * s_size;

        for (size_t s = 0; s < sites; ++s) {
            value_t* x = &data.get()[hour * h_size + s * s_size + 1];

            for (size_t h = hour; h < 2160; ++h) {
                x[1] = h - hour;
                value_t p = optimize::predict(&theta[0], x, features_size);
                predictions[h * sites + s] = p;
            }
        }
    }


    std::vector<double> forecast(int hour, const std::vector<int>& data, double K, const std::vector<double>& globalQuakes) {
        size_t return_size = sites * 2160;
        std::vector<double> predictions(return_size, 0.);

        for (size_t s = 0; s < sites; ++s) {
            size_t beg = s * rate * 3600 * 3;
            size_t size = rate * 3600 * 3;

            add_observation(hour, s, &data[beg], size, globalQuakes);
        }

        fit_classifier(hour);
        predict(hour, predictions);

        return predictions;
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



void test5() {
    size_t rate = 30;
    size_t sites = 2;

    std::vector<int> data;
    std::vector<double> locations;
    std::vector<double> quakes;

    //
    locations.push_back(37.6156);
    locations.push_back(55.7522);
    locations.push_back(103.8);
    locations.push_back(1.3667);

    quakes.push_back(37.);
    quakes.push_back(55.);
    quakes.push_back(1);
    quakes.push_back(3);
    quakes.push_back(0);

    for (size_t s = 0; s < sites; ++s) {
        for (size_t r = 0; r < (rate * 3600 * 3); ++r) {
            data.push_back((int)(100. * rnd::rand()));
        }
    }

    QuakePredictor qp;
    {
        qp.init(rate, sites, locations);
 
        std::vector<double> p1 = qp.forecast(0, data, 2, quakes);
        std::vector<double> p2 = qp.forecast(1, data, 3, quakes);
    
        std::cout << "p1 size: " << p1.size() << std::endl;
        std::cout << "p2 size: " << p2.size() << std::endl;

    }

    std::cout << "test5 done" << std::endl;
}


int main() {

    test();
    test2();
    test3();
    test4();
    test5();

    return 0;
}


#endif

















