#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_transition_entropy_signret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<int> s;
            s.reserve(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = std::log(p1) - std::log(p0);
                if (!std::isfinite(r)) continue;
                int sg = (r > 0.0f) ? 1 : (r < 0.0f ? -1 : 0);
                if (sg != 0) s.push_back(sg);
            }
            if (s.size() < 2) { res_ptr[i*d1+j] = std::numeric_limits<float>::quiet_NaN(); continue; }
            double tr[2][2] = {{0.0,0.0},{0.0,0.0}};
            for (size_t k=1;k<s.size();++k){
                int a = s[k-1]>0 ? 1 : 0;
                int b = s[k]>0 ? 1 : 0;
                tr[a][b] += 1.0;
            }
            double rowsum[2] = {tr[0][0]+tr[0][1], tr[1][0]+tr[1][1]};
            double total = rowsum[0] + rowsum[1];
            if (total == 0.0) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double H = 0.0;
            for (int r=0;r<2;++r){
                if (rowsum[r] == 0.0) continue;
                for (int c=0;c<2;++c){
                    if (tr[r][c] > 0.0){
                        double p = tr[r][c]/rowsum[r];
                        H += (rowsum[r]/total)*(-p*std::log2(p));
                    }
                }
            }
            res_ptr[i*d1+j] = static_cast<float>(H); // normalized by log2(2)=1
        }
    }
}

void bind_cube2mat_transition_entropy_signret(py::module& m) {
    m.def("cube2mat_transition_entropy_signret", &cube2mat_transition_entropy_signret,
          py::arg("result"), py::arg("cubes_map"),
          "Conditional entropy H(S_t | S_{t-1}) of sign(log returns), normalized by log2(2)=1.");
}

