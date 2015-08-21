

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>



using namespace std;



const size_t train_rows = 32;
size_t train_cols = 0;

const size_t events_rows = 6;
size_t events_cols = 0;




namespace dft {

double pi = 4 * atan(1.0);

void fft(int sign, vector<complex<double> > &zs) {
    unsigned int j=0;
    // Warning about signed vs unsigned comparison
    for(unsigned int i=0; i<zs.size()-1; ++i) {
        if (i < j) {
            complex<double> t = zs[i];
            zs[i] = zs[j];
            zs[j] = t;
        }
        int m=zs.size()/2;
        j^=m;
        while ((j & m) == 0) { m/=2; j^=m; }
    }
    for(unsigned int j=1; j<zs.size(); j*=2)
        for(unsigned int m=0; m<j; ++m) {
            double t = pi * sign * m / j;
            complex<double> w(::cos(t), ::sin(t));
            for(unsigned int i = m; i<zs.size(); i+=2*j) {
                complex<double> zi = zs[i], t = w * zs[i + j];
                zs[i] = zi + t;
                zs[i + j] = zi - t;
            }
        }
}


}





float* read(const char* fname, size_t& size_floats) {
    FILE* fd = fopen(fname, "rb");
    if (fd) {
        fseek(fd, 0, SEEK_END);
        size_t size_bytes = ftell(fd);
        fseek(fd, 0, SEEK_SET);
        size_floats = size_bytes / sizeof(float);
       
        float* tmp = new float[size_floats];
        fread(tmp, size_floats * sizeof(float), 1, fd); 
        fclose(fd);


        cout << "from " << fname << " read " << size_floats << " float values" << endl;
   
        return tmp;
    }

    return NULL;
}







void vectorize(float* data, float* events, size_t data_rows, size_t data_cols, size_t events_rows, size_t events_cols, const char* fn_out, const size_t step) {
    FILE* fout = fopen(fn_out, "wb+");

    vector<double> tmp;
    tmp.reserve(1024 * 30);

    size_t total_rows = 0;
    size_t total_cols = 0;

    if (fout) {
        const size_t rate = 500;
        const size_t width = 512 / 2;


        // filter out everything after 40 Hz
        size_t f_end_idx = int(40. * width / rate);
        size_t N = f_end_idx;



        size_t beg = 0;
        size_t end = beg + width;



        vector<complex<double> > F;
        F.reserve(width);


        while(end <= data_cols) {

            tmp.clear();

            size_t ROW_SIZE = 0;

            for (size_t row = 0; row < data_rows; ++row) {

                F.clear();

                double n, M1, M2, M3, M4;
                double delta, delta_n, delta_n2, term1;
                // init
                n = M1 = M2 = M3 = M4 = 0.;

                for (size_t i = 0; i < width; ++i) {
                    // mvsk
                    double n1 = n++;
                    delta = double(data[row*data_cols + beg + i]) - M1;
                    delta_n = delta / n;
                    delta_n2 = delta_n * delta_n;
                    term1 = delta * delta_n * n1;
                    M1 += delta_n;
                    M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
                    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
                    M2 += term1;

                    // prepare vector for FFT at the same time
                    F.push_back(complex<double>(data[row*data_cols + beg + i]));
                }

                double m = M1;
                double v = M2 / (n - 1.0);
                double s = ::sqrt(n) * M3 / ::pow(M2, 1.5);
                double k = n * M4 / (M2 * M2) - 3.0;

                tmp.push_back(m);
                tmp.push_back(v);
                tmp.push_back(s);
                tmp.push_back(k);
                ROW_SIZE += 4;

                dft::fft(-1, F);

                double gen_power = 0.;

                for (size_t f = 0; f < N; ++f) {
                    tmp.push_back(F[f].real());
                    ++ROW_SIZE;
                    tmp.push_back(F[f].imag());
                    ++ROW_SIZE;
                    // magnitude
                    tmp.push_back(::sqrt(F[f].real()*F[f].real() + F[f].imag()*F[f].imag()) / N);
                    ++ROW_SIZE;
                    // power
                    tmp.push_back(F[f].real()*F[f].real());
                    ++ROW_SIZE;
                    //
                    double r = F[f].real();
                    r = r == 0. ? 0.0000001 : r;
                    tmp.push_back(::atan(F[f].imag() / r));
                    ++ROW_SIZE;
                    //
                    gen_power += F[f].real() * F[f].real();
                    //
                }  
                tmp.push_back(gen_power);
                ++ROW_SIZE;

            } // for row

            // pairwise cross correlations
            ROW_SIZE /= data_rows;
            size_t MEAN_IDX = 0;
            size_t VAR_IDX = 1;
            for (size_t i = 0; i < data_rows; ++i) {
                for (size_t j = i+1; j < data_rows; ++j) {
                    double cor = 0.;
                    for (size_t cc = 0; cc < width; ++cc) {
                        cor += (data[i*data_cols + beg + cc] - tmp[i*ROW_SIZE + MEAN_IDX]) *
                               (data[j*data_cols + beg + cc] - tmp[j*ROW_SIZE + MEAN_IDX]); 
                    }
                    cor /= width;
                    cor /= ::sqrt(tmp[i*ROW_SIZE + VAR_IDX]) * ::sqrt(tmp[j*ROW_SIZE + VAR_IDX]);
                    tmp.push_back(cor);
                }
            }

            // add Ys - 6 event channels
            for (size_t row = 0; row < events_rows; ++row) {
                double e = events[row*events_cols + beg + width - 1];
                tmp.push_back(e);
            }


            //
            beg += step;
            end = beg + width;

            fwrite(&tmp[0], tmp.size() * sizeof(double), 1, fout);

            ++total_rows;
            total_cols = tmp.size();
        }
    }

    fclose(fout);

    cout << "Rows / Cols: " << total_rows << " " << total_cols << endl;
}






int main(int argc, const char* argv[]) {

    const char* fn_data = argv[1];
    const char* fn_events = argv[2];
    const char* fn_out = argv[3];
    const size_t step = ::atoll(argv[4]);

    size_t size_floats = 0;

    float* data = read(fn_data, size_floats);
    train_cols = size_floats / train_rows;

    float* events = read(fn_events, size_floats);
    events_cols = size_floats / events_rows;


   
    vectorize(data, events, train_rows, train_cols, events_rows, events_cols, fn_out, step);



    // free memory
    delete [] data;
    delete [] events;


    return 0;
}
