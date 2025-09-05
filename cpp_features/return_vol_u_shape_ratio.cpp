#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static bool compute_std(const std::vector<float>& v, float& out) {
    size_t n = v.size();
    if (n < 2) return false;
    double mean = 0.0; for (float x : v) mean += x; mean /= n;
    double var = 0.0; for (float x : v) { double d = x - mean; var += d*d; }
    var /= (n - 1);
    out = static_cast<float>(std::sqrt(var));
    return true;
}

void cube2mat_return_vol_u_shape_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                       const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1)
        throw std::runtime_error("result must be (d0,d1)");

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    // indices in returns vector for mid-period 11:00-14:00 (36 returns) assuming 5-min bars
    const size_t mid_start = 18; // t index 19 in returns (11:00->11:05)
    const size_t mid_end = 53;   // inclusive index corresponding to 14:00 return

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> rets;
            rets.reserve(d2);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (p0 > 0.f && p1 > 0.f && !std::isnan(p0) && !std::isnan(p1))
                    rets.push_back(std::log(p1 / p0));
            }
            size_t n = rets.size();
            if (n < 54) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // First hour std (first 12 returns)
            std::vector<float> first_hour(rets.begin(), rets.begin()+12);
            std::vector<float> last_hour(rets.end()-12, rets.end());
            if (mid_end >= n) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<float> mid_period(rets.begin()+mid_start, rets.begin()+mid_end+1);
            float sd_first, sd_last, sd_mid;
            if (!compute_std(first_hour, sd_first) || !compute_std(last_hour, sd_last) ||
                !compute_std(mid_period, sd_mid) || !(sd_mid>0.f)) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            float numerator = (sd_first + sd_last) * 0.5f;
            res_ptr[i*d1 + j] = numerator / sd_mid;
        }
    }
}

void bind_cube2mat_return_vol_u_shape_ratio(py::module& m) {
    m.def("cube2mat_return_vol_u_shape_ratio", &cube2mat_return_vol_u_shape_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "U-shape volatility ratio: avg(std first hour, last hour) / std mid-day.");
}

