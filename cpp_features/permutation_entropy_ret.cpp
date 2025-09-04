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

static int perm_index(const std::vector<int>& rank) {
    int m = rank.size();
    int idx = 0;
    std::vector<int> used(m,0);
    for (int i=0;i<m;++i) {
        int r = rank[i];
        int cnt=0;
        for (int v=0; v<r; ++v) if (!used[v]) ++cnt;
        idx = idx*(m - i) + cnt;
        used[r]=1;
    }
    return idx;
}

void cube2mat_permutation_entropy_ret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    const int m = 4;
    const int tau = 1;
    const double norm = std::log(std::tgamma(m + 1.0));

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
            int L = n - (m - 1) * tau;
            if (L <= 1) { res_ptr[i*d1+j] = std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> counts(24, 0.0);
            std::array<std::pair<float,int>,4> arr;
            std::vector<int> rank(m);
            for (int idx=0; idx<L; ++idx) {
                for (int k=0;k<m;++k) arr[k]={r[idx + k*tau], k};
                std::stable_sort(arr.begin(), arr.end(), [](auto& a, auto& b){return a.first < b.first;});
                for (int k=0;k<m;++k) rank[arr[k].second]=k;
                int key = perm_index(rank);
                counts[key] += 1.0;
            }
            double total=0.0; for(double c:counts) total+=c;
            if (total <= 0.0) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double H=0.0;
            for(double c:counts) if(c>0){ double p=c/total; H -= p*std::log(p);} 
            res_ptr[i*d1+j]=static_cast<float>(H / norm);
        }
    }
}

void bind_cube2mat_permutation_entropy_ret(py::module& m) {
    m.def("cube2mat_permutation_entropy_ret", &cube2mat_permutation_entropy_ret,
          py::arg("result"), py::arg("cubes_map"),
          "Normalized permutation entropy of log returns (m=4, tau=1).");
}

