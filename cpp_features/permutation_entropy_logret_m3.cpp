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

static int pattern_index(const float* w) {
    std::array<std::pair<float,int>,3> arr{std::make_pair(w[0],0), std::make_pair(w[1],1), std::make_pair(w[2],2)};
    std::stable_sort(arr.begin(), arr.end(), [](auto& a, auto& b){return a.first < b.first;});
    int order0=arr[0].second, order1=arr[1].second, order2=arr[2].second;
    if (order0==0 && order1==1 && order2==2) return 0;
    if (order0==0 && order1==2 && order2==1) return 1;
    if (order0==1 && order1==0 && order2==2) return 2;
    if (order0==1 && order1==2 && order2==0) return 3;
    if (order0==2 && order1==0 && order2==1) return 4;
    return 5; // 2,1,0
}

void cube2mat_permutation_entropy_logret_m3(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                            const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto close_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto cbuf = close_arr.request();
    if (cbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* close_ptr = static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const int m = 3;
    const double norm = std::log2(6.0);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> r;
            r.reserve(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float ret = std::log(p1) - std::log(p0);
                if (std::isfinite(ret)) r.push_back(ret);
            }
            int n = r.size();
            if (n < m) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double counts[6]={0};
            for (int idx=0; idx<=n-m; ++idx) {
                int k = pattern_index(&r[idx]);
                counts[k] += 1.0;
            }
            double total=0.0; for (double c : counts) total += c;
            if (total <= 0.0) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double H=0.0;
            for (double c : counts) if (c>0) { double p=c/total; H -= p*std::log2(p);} 
            res_ptr[i*d1+j]=static_cast<float>(H/norm);
        }
    }
}

void bind_cube2mat_permutation_entropy_logret_m3(py::module& m) {
    m.def("cube2mat_permutation_entropy_logret_m3", &cube2mat_permutation_entropy_logret_m3,
          py::arg("result"), py::arg("cubes_map"),
          "Permutation entropy (m=3) of RTH log returns, normalized to [0,1].");
}

