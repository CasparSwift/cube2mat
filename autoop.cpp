#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <functional>


namespace py = pybind11;


float compute_nanmean(const float* data, size_t n) {
    size_t count = 0;
    float sum = 0.0;
    // 第一遍：计算非 NaN 数据的和与计数
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (!std::isnan(v)) {
            sum += v;
            count++;
        }
    }
    if (count == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float mean = sum / count;
    return mean;
}


// 一元cube2mat function 模板
template <typename Func>
void fast_apply_cube_func(py::array_t<float>& result, const py::array_t<float>& cube, Func cube2mat_func) {
    auto cube_buf = cube.request();
    if (cube_buf.ndim != 3) {
        throw std::runtime_error("Input cube must be 3-dimensional");
    }
    ssize_t d0 = cube_buf.shape[0];
    ssize_t d1 = cube_buf.shape[1];
    ssize_t d2 = cube_buf.shape[2];

    auto res_buf = result.request();
    if (res_buf.ndim != 2 || res_buf.shape[0] != d0 || res_buf.shape[1] != d1) {
        throw std::runtime_error("Result array must have shape (cube.shape[0], cube.shape[1])");
    }

    float* cube_ptr = static_cast<float*>(cube_buf.ptr);
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    #pragma omp parallel for collapse(2) schedule(static)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            float* slice_ptr = cube_ptr + i * (d1 * d2) + j * d2;
            res_ptr[i * d1 + j] = cube2mat_func(slice_ptr, d2);
        }
    }
}

// cube2mat function definition
void tsmeanfast(py::array_t<float>& result, const py::array_t<float>& cube) {
    fast_apply_cube_func(result, cube, compute_nanmean);
}



PYBIND11_MODULE(op_modules, m) {
    m.doc() = "pybind11 example plugin: fast_operation implementation";
    m.def("transpose_cube", &transpose_cube, "Transpose a 3D cube with axes order (0,2,1)");
    m.def("tsmeanfast", &tsmeanfast);
    m.def("tsstddevfast", &tsstddevfast);
    m.def("tsirfast", &tsirfast);
    m.def("tspercentagefast", &tspercentagefast);
    m.def("tsmedianfast", &tsmedianfast);
    m.def("tsiqrmeanfast", &tsiqrmeanfast);
    m.def("tsiqrmedianfast", &tsiqrmedianfast);
    m.def("tskurtosisfast", &tskurtosisfast);
    m.def("tsskewnessfast", &tsskewnessfast);
    m.def("tsautocorrfast", &tsautocorrfast);
    m.def("tsslopefast", &tsslopefast);
    
    m.def("tscorrfast", &tscorrfast);
    m.def("tsregressionfast", &tsregressionfast);

    m.def("groupmeanfast", &groupmeanfast<float>);
    m.def("groupminfast", &groupminfast<float>);
    m.def("groupmaxfast", &groupmaxfast<float>);
    m.def("groupsumfast", &groupsumfast<float>);
    m.def("groupstddevfast", &groupstddevfast<float>);
    m.def("groupmedianfast", &groupmedianfast<float>);
    m.def("grouprankfast", &grouprankfast<float>);
    m.def("groupneutfast", &groupneutfast<float>);
    m.def("groupprojfast", &groupprojfast<float>);
    m.def("tsrankfast_daily", &tsrankfast_daily<float>);
    m.def("rankfast", &rankfast<float>);
    m.def("transpose_120", &transpose_120<float>, 
        py::arg("input").noconvert(), 
        py::arg("output").noconvert(),
        "In-place transpose 3D array with axes (1,2,0)");
}
