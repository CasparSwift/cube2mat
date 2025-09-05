#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_rv_front_loading_score(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map){
    if(!cubes_map.contains("last_price")){
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf = result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> r2; r2.reserve(d2);
            std::vector<double> xs; xs.reserve(d2);
            double total=0.0;
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(std::isnan(p0)||std::isnan(p1)||!(p0>0.f)||!(p1>0.f)) continue;
                double r=std::log(static_cast<double>(p1)/p0);
                if(!std::isfinite(r)) continue;
                double r2v=r*r;
                total+=r2v;
                r2.push_back(r2v);
                double tf=(d2>1)?static_cast<double>(t)/static_cast<double>(d2-1):0.0;
                xs.push_back(tf);
            }
            if(r2.empty() || !(total>0.0)){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> ys; ys.reserve(r2.size());
            double cum=0.0;
            for(size_t k=0;k<r2.size();++k){
                cum+=r2[k];
                ys.push_back(cum/total);
            }
            if(xs.front()>0.0){ xs.insert(xs.begin(),0.0); ys.insert(ys.begin(),0.0); }
            if(xs.back()<1.0){ xs.push_back(1.0); ys.push_back(1.0); }
            double auc=0.0;
            for(size_t k=1;k<xs.size();++k){
                double dx=xs[k]-xs[k-1];
                double av=(ys[k]+ys[k-1])/2.0;
                auc+=av*dx;
            }
            double score=2.0*auc-1.0;
            if(score>1.0) score=1.0;
            if(score<-1.0) score=-1.0;
            res_ptr[i*d1+j]=static_cast<float>(score);
        }
    }
}

void bind_cube2mat_rv_front_loading_score(py::module& m){
    m.def("cube2mat_rv_front_loading_score", &cube2mat_rv_front_loading_score,
          py::arg("result"), py::arg("cubes_map"),
          "2*AUC(cum RV fraction vs time fraction)-1 within RTH.");
}

