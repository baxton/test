
// utils.cpp
//
// g++ -std=c++11 utils.cpp -shared -o utils.dll
//



#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>


using namespace std;



extern "C" {



    void get_statistics(const double* vec, int size, double* mean, double* skewness, double* variance, double* kurtosis) {
        double n, M1, M2, M3, M4;
        double delta, delta_n, delta_n2, term1;
        // init
        n = M1 = M2 = M3 = M4 = 0.;

        for (size_t i = 0; i < size; ++i) {
            // mvsk
            double n1 = n++;
            delta = double(vec[i]) - M1;
            delta_n = delta / n;
            delta_n2 = delta_n * delta_n;
            term1 = delta * delta_n * n1;
            M1 += delta_n;
            M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
            M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
            M2 += term1;
        }

        double m = M1;                                 // mean
        double v = M2 / (n - 1.0);                     // variance
        double s = ::sqrt(n) * M3 / ::pow(M2, 1.5);    // skewness
        double k = n * M4 / (M2 * M2) - 3.0;           // kurtosis

    
        if (mean) 
            *mean = m;
        if (skewness)
            *skewness  = s;
        if (variance)
            *variance = v;
        if (kurtosis)
            *kurtosis = k;
    }


    void get_frequencies(const double* frames, 
                         int rows, 
                         int cols, 
                         int frames_num, 
                         double* freq,
                         double MEAN_MUL,
                         double LOW_VAL,
                         double HIGH_VAL
                         ) {
        size_t frame_size = rows * cols;
        const double* prev = frames;

        unique_ptr<double[]> deltas(new double[frame_size]);

        for (int f = 1; f < frames_num; ++f) {
            const double* curr = &prev[frame_size];

            // get delta
            for (int p = 0; p < frame_size; ++p) {
                deltas[p] = (prev[p] - curr[p]) * (prev[p] - curr[p]);
            }

            double mean, std;
            get_statistics(deltas.get(), frame_size, &mean, nullptr, &std, nullptr);
            std = sqrt(std);

            // tmp[zz < mv + std*MEAN_MUL] = 0
            // freq[tmp != 0] += 1
            for (int p = 0; p < frame_size; ++p) {
                double deviation = deltas[p] - mean;
                if (deviation >= (mean + std * MEAN_MUL)) {
                    freq[p] += 1.;
                }
            }
            
            // prepare for the next iteration 
            prev = curr;  
        }

        for (int p = 0; p < frame_size; ++p) {
            if (freq[p] < LOW_VAL || HIGH_VAL < freq[p])
                freq[p] = 0;
        }
    }



    void filter_frequencies() {
    for e in range(EPOCHES):
        R = shape2D[0]
        C = shape2D[1]
        S = FRAME_SIZE
        K = KOEF
        res = np.zeros(shape2D, dtype=float)
        for r in range(R):
            if (r + S) >= R:
                continue
            for c in range(C):
                if (c + S) >= C:
                    continue
                tmp = freq[r : r + S, c : c + S]
                m = np.mean(tmp)

                if K < m:
                    res[r,c] = 100
                else:
                    res[r,c] = 0

        beg_row = S/2
        neg_beg_row = freq.shape[0] - (freq.shape[0] - beg_row)
        beg_col = S/2
        neg_beg_col = freq.shape[1] - (freq.shape[1] - beg_col)
        freq[beg_row :, beg_col :] = res[:-neg_beg_row, :-neg_beg_col]
        freq[:beg_row, :] = 0
        freq[:, :beg_col] = 0
    }


}
