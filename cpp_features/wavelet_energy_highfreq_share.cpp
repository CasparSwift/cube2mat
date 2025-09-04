#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static bool haar_approx(std::vector<double> a, int L, std::vector<double>& out) {
    for (int lvl = 0; lvl < L; ++lvl) {
        size_t n = a.size();
        if (n < 2) return false;
        if (n % 2 == 1) { a.push_back(a.back()); ++n; }
        std::vector<double> next(n/2);
        for (size_t i = 0; i < n/2; ++i) {
            next[i] = (a[2*i] + a[2*i+1]) / std::sqrt(2.0);
        }
        a.swap(next);
    }
    out = a;
    return true;
}

static void detrend_series(const std::vector<double>& y, std::vector<double>& e) {
    size_t n = y.size();
    double sum_t=0.0,sum_y=0.0,sum_t2=0.0,sum_ty=0.0;
    for (size_t i=0;i<n;++i){double t=i/(double)(n-1);double yy=y[i];sum_t+=t;sum_y+=yy;sum_t2+=t*t;sum_ty+=t*yy;}
    double den=n*sum_t2-sum_t*sum_t;double a=(sum_y*sum_t2-sum_t*sum_ty)/den;double b=(n*sum_ty-sum_t*sum_y)/den;
    e.resize(n);double mean=0.0;for(size_t i=0;i<n;++i){double t=i/(double)(n-1);e[i]=y[i]-(a+b*t);mean+=e[i];}
    mean/=n;for(size_t i=0;i<n;++i)e[i]-=mean;
}

void cube2mat_wavelet_energy_highfreq_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                            const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const int L = 3;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> y;
            y.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (!std::isnan(c)) y.push_back(static_cast<double>(c));
            }
            size_t n = y.size();
            if (n < 8) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> e; detrend_series(y,e);
            double tot=0.0; for(double v:e) tot+=v*v; if(!(tot>0)){res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue;}
            std::vector<double> aL; if(!haar_approx(e,L,aL) || aL.empty()){res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue;}
            double low=0.0; for(double v:aL) low+=v*v;
            double val=1.0 - low/tot; if(val<0) val=0; if(val>1) val=1;
            res_ptr[i*d1+j]=static_cast<float>(val);
        }
    }
}

void bind_cube2mat_wavelet_energy_highfreq_share(py::module& m) {
    m.def("cube2mat_wavelet_energy_highfreq_share", &cube2mat_wavelet_energy_highfreq_share,
          py::arg("result"), py::arg("cubes_map"),
          "Haar-DWT high-frequency energy share of detrended close (L=3).");
}

