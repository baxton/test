
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

#if defined(TESTS) || defined(EMUL)
#    include <fstream>
#    include <iostream>
#    include <string>
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
    linalg::mul_scalar(delta, x, grad_x, columns);

    return cost;
}


void minimize_gc(value_t* theta, const value_t* x, int columns, const value_t y, int max_iterations) {
    value_t grad[columns];

    value_t e = 0.0001;
    value_t a = .8;

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


const size_t theta_num = 10;

class QuakePredictor {


    size_t rate;
    size_t sites;
    std::vector<value_t> sitesLocations;

    size_t columns;
    std::vector<value_t> theta[theta_num];
    std::vector<value_t> theta2;


    size_t features_size;
    std::auto_ptr<value_t> data;

    std::vector<double> predictions;


    QuakePredictor(const QuakePredictor&);
    QuakePredictor& operator=(const QuakePredictor&);

public:
    QuakePredictor() :
        rate(0),
        sites(0),
        sitesLocations(),
        columns(0),
        //theta(),
        features_size(0),
        data(NULL),
        predictions()
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
        value_t MAX_DIST = 1000.;

        size_t count = 0;
        size_t size = quakes.size() / 5;

        value_t score = 0.;

        for (size_t i = 0; i < size; ++i) {
            value_t dist = distance(lat, lon, quakes[i * 5], quakes[i * 5 + 1]);
#if defined EMUL
log() << ">>" << dist << std::endl;
#endif
            count += dist <= MAX_DIST;

            score += .75 * (50. / dist)*(50. / dist) ; //+ .05 *  quakes[i * 5 + 3] + 1. / quakes[i * 5 + 2] + .01 * (dist <= MAX_DIST);
        }
#if defined EMUL
std::cerr << ">>" << (size ? score / size : 0.) << std::endl;
#endif
        return (size ? score / size : 0.);
    }


    int init(int sampleRate, int numOfSites, const std::vector<double>& sitesData) {
        rate = sampleRate;
        sites = numOfSites;
        std::copy(sitesData.begin(), sitesData.end(), std::back_inserter(sitesLocations));


        // allocate memory for data
        // Format:
        // [hour #][site #][Y + <17 features>]
        // 17 features:
        //    <always 1><hour><hour^2><hour^3><activity><mvsk * 3>
        features_size = 17;
        size_t total_buffer_size = 2160 * sites * (1 + features_size);
        data = std::auto_ptr<value_t>(new value_t[total_buffer_size]);


        size_t return_size = sites * 2160;
        for (size_t i = 0; i < return_size; ++i)
            predictions.push_back(0.);


        // init classifier
        for (size_t i = 0; i < features_size; ++i) {
            for (size_t t = 0; t < theta_num; ++t) {
                theta[t].push_back(rnd::rand() );
            }
        }

        return 0;
    }

    void add_observation(size_t hour, size_t site, const int* vec, size_t size, value_t event, value_t K) {
        size_t s_size = (1 + features_size);
        size_t h_size = sites * s_size;
        value_t* features = &data.get()[hour * h_size + site * s_size];

        features[0] = event;
        features[1] = 1.;   // always 1.
        //features[2] = time 
        //features[3] = time^2
        //features[4] = time^3
        features[5] = K;

        size_t beg = 0;
        size_t vec_size = size / 3;

        size_t offset = 6;

        stat::get_stat_online(&vec[beg], vec_size, features[offset + 0], features[offset + 1], features[offset + 2], features[offset + 3]);
        beg += vec_size;
        stat::get_stat_online(&vec[beg], vec_size, features[offset + 4], features[offset + 5], features[offset + 6], features[offset + 7]);
        beg += vec_size;
        stat::get_stat_online(&vec[beg], vec_size, features[offset + 8], features[offset + 9], features[offset + 10], features[offset + 11]);
    }



    void fit_classifier(size_t hour) {
        size_t s_size = (1 + features_size);
        size_t h_size = sites * s_size;


        for (size_t h = 0; h <= hour; ++h) {
            for (size_t s = 0; s < sites; ++s) {
                value_t y = data.get()[hour * h_size + s * s_size];   // TODO move out of the loop
                value_t* x = &data.get()[h * h_size + s * s_size + 1];

                value_t t = value_t(hour - h) / 2160.; 
                x[1] = t;
                x[2] = t * t;
                x[3] = t * t * t;

                for (size_t t = 0; t < theta_num; ++t)
                    optimize::minimize_gc(&theta[t][0], x, features_size, y, 1);
            }
        }
    }



    void predict(size_t hour) {
        size_t s_size = (1 + features_size);
        size_t h_size = sites * s_size;

        for (size_t s = 0; s < sites; ++s) {
            value_t* x = &data.get()[hour * h_size + s * s_size + 1];

            for (size_t h = hour; h < 2160; ++h) {
                value_t t = value_t(h - hour) / 2160;
                x[1] = t;
                x[2] = t * t;
                x[3] = t * t * t;

                value_t p = 0.;
                for (size_t t = 0; t < theta_num; ++t) {
                    value_t tmp = optimize::predict(&theta[t][0], x, features_size);
                    p += tmp;
                }
 
                p /= theta_num;
                value_t mul = 1.; //predictions[h * sites + s] > p ? .9 : 1.;
                predictions[h * sites + s] = p * mul;
            }
        }
    }


    std::vector<double> forecast(int hour, const std::vector<int>& data, double K, const std::vector<double>& globalQuakes) {

        if (0 < globalQuakes.size()) {

            std::vector<value_t> scores;
            for (size_t s = 0; s < sites; ++s) {
                scores.push_back(is_event(sitesLocations[s*2], sitesLocations[s*2 + 1], globalQuakes));
            }
            value_t m, v, s, k;
            stat::get_stat_online(&scores[0], sites, m, v, s, k);
#if defined EMUL
std::cerr << "mvsk: " << m << " " << v << " " << s << " " << k << std::endl;
#endif

            if (s <= 2. && -2. <= k && k <= 2. && !(1.8 <= s && s <= 2.1 && 1.8 <= k && k <= 2.1)) {

                for (size_t s = 0; s < sites; ++s) {
                    size_t beg = s * rate * 3600 * 3;
                    size_t size = rate * 3600 * 3;

                    value_t event = scores[s] < (m > .0002 ? .0001 : m * 3) ? 1. : 0.;
#if defined EMUL
log() << "hour " << hour << " site " << s << " [" << sitesLocations[s*2] << ", " << sitesLocations[s*2
+1] << "]" << " event " << event << std::endl;
#endif
                    add_observation(hour, s, &data[beg], size, event, K);
                }

                size_t epoches = hour < 10 ? 10 : 50;
                if (0 <= s && s <= 1.6 && .5 <= k && k <= 1.9)
                    epoches = 150;

                for (size_t e = 0; e < epoches; ++e) {
                    for (size_t h = 0; h <= hour; ++h)
                        fit_classifier(h);
                }

            }
        }
        predict(hour);

        return predictions;
    }



    //
    // for debugging
    //

//    const std::vector<value_t>& get_theta() { return theta; }
#if defined EMUL
    static std::ostream& log() {
        static std::ofstream f("quake.log", std::ios_base::out | std::ios_base::trunc);

        return f;
    }
#endif
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




// ----------------------------------------------
// Running under emulator
// ----------------------------------------------


#if defined EMUL

int main() {

    std::string line;

    std::getline(std::cin, line);
    int rate = std::atoi(line.c_str());

    std::getline(std::cin, line);
    int S = std::atoi(line.c_str());

    std::getline(std::cin, line);
    int SLEN = std::atoi(line.c_str());

    std::vector<double> sitesData;
    for (int i = 0; i < SLEN; ++i) {
        std::getline(std::cin, line);
        double d = std::atof(line.c_str());
        sitesData.push_back(d);
    }



    QuakePredictor qp;
    int ret = qp.init(rate, S, sitesData);
    std::cout << ret << std::endl;


    std::getline(std::cin, line);
    int doTraining = std::atoi(line.c_str());

    if (1 == doTraining) {
        ;
    }


    while(std::getline(std::cin, line)) {
        int hour = std::atoi(line.c_str());
        if (-1 == hour) break;

        std::getline(std::cin, line);
        int DLEN = std::atoi(line.c_str());

        std::vector<int> data;
        for (int i = 0; i < DLEN; ++i) {
            std::getline(std::cin, line);
            int d = std::atoi(line.c_str());
            data.push_back(d);
        }

        std::getline(std::cin, line);
        int K = std::atoi(line.c_str());

        std::getline(std::cin, line);
        int QLEN = std::atoi(line.c_str());

        std::vector<double> globalQuakes;
        for (int i = 0; i < QLEN; ++i) {
            std::getline(std::cin, line);
            double d = std::atof(line.c_str());
            globalQuakes.push_back(d);
        }


        std::vector<double> retM = qp.forecast(hour, data, K, globalQuakes);
        std::cout << retM.size() << std::endl;

        for (int i = 0; i < retM.size(); ++i) {
            std::cout << retM[i] << std::endl;
        }
    }

    return 0;
}





#endif









