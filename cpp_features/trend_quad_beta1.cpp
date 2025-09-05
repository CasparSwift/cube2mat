#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static bool solve_quad(const std::vector<float>& y, double& b1){
    size_t n=y.size();
    if(n<3) return false;
    double sum_t=0.0, sum_t2=0.0, sum_t3=0.0, sum_t4=0.0;
    double sum_y=0.0, sum_ty=0.0, sum_t2y=0.0;
    double denom = (n>1)?(n-1):1; // to compute t=k/(n-1)
    for(size_t k=0;k<n;++k){
        double t = denom>0? (k/denom):0.0;
        double t2 = t*t;
        sum_t += t;
        sum_t2 += t2;
        sum_t3 += t2*t;
        sum_t4 += t2*t2;
        double yv = y[k];
        sum_y += yv;
        sum_ty += t*yv;
        sum_t2y += t2*yv;
    }
    double A[3][3]={{(double)n,sum_t,sum_t2},{sum_t,sum_t2,sum_t3},{sum_t2,sum_t3,sum_t4}};
    double B[3]={sum_y,sum_ty,sum_t2y};
    double det =
        A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) -
        A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) +
        A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if(std::fabs(det) < 1e-12) return false;
    // determinant for beta1 column
    double A1[3][3]={{B[0],A[0][1],A[0][2]},{B[1],A[1][1],A[1][2]},{B[2],A[2][1],A[2][2]}};
    double det1 =
        A1[0][0]*(A1[1][1]*A1[2][2]-A1[1][2]*A1[2][1]) -
        A1[0][1]*(A1[1][0]*A1[2][2]-A1[1][2]*A1[2][0]) +
        A1[0][2]*(A1[1][0]*A1[2][1]-A1[1][1]*A1[2][0]);
    b1 = det1/det;
    return true;
}

void cube2mat_trend_quad_beta1(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> y; y.reserve(d2);
            for(ssize_t t=0;t<d2;++t){ float c=price_ptr[base+t]; if(!std::isnan(c)) y.push_back(c); }
            double b1;
            if(!solve_quad(y,b1)) res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            else res_ptr[i*d1+j]=static_cast<float>(b1);
        }
    }
}

void bind_cube2mat_trend_quad_beta1(py::module& m){
    m.def("cube2mat_trend_quad_beta1", &cube2mat_trend_quad_beta1,
          py::arg("result"), py::arg("cubes_map"),
          "Linear coefficient (beta1) of quadratic trend close~1+t+t^2 in RTH.");
}

