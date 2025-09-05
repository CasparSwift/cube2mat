#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float median(std::vector<float>& v) {
    if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
    size_t n = v.size();
    size_t mid = n/2;
    std::nth_element(v.begin(), v.begin()+mid, v.end());
    float med = v[mid];
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin()+mid-1, v.end());
        med = 0.5f*(med + v[mid-1]);
    }
    return med;
}

void cube2mat_jump_count_local_k3_win21(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                        const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const int W = 21;
    const int HW = W/2; // 10
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i*(d1*d2) + j*d2;
            std::vector<float> logret(d2-1, std::numeric_limits<float>::quiet_NaN());
            for (ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if (!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                logret[t-1]=std::log(p1)-std::log(p0);
            }
            int cnt=0;
            ssize_t n=logret.size();
            for(ssize_t t=0;t<n;++t){
                if(!std::isfinite(logret[t])) continue;
                ssize_t s= (t>HW)? t-HW:0;
                ssize_t e= std::min<ssize_t>(n-1, t+HW);
                std::vector<float> win; win.reserve(e-s+1);
                for(ssize_t k=s;k<=e;++k) if(std::isfinite(logret[k])) win.push_back(logret[k]);
                if(win.size()<2) continue;
                float med = median(win);
                for(float& x:win) x = std::fabs(x - med);
                float mad = median(win);
                float scale = 1.4826f * mad;
                if(!(scale>0.f)) continue;
                if (std::fabs(logret[t]) > 3.0f * scale) ++cnt;
            }
            res_ptr[i*d1+j]=static_cast<float>(cnt);
        }
    }
}

void bind_cube2mat_jump_count_local_k3_win21(py::module& m){
    m.def("cube2mat_jump_count_local_k3_win21", &cube2mat_jump_count_local_k3_win21,
          py::arg("result"), py::arg("cubes_map"),
          "Count of |logret|>3×local MAD-scale events (window=21).");
}

