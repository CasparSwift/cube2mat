#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_alpha_example(py::module& m);
void bind_cube2mat_gini_absret(py::module& m);
void bind_cube2mat_n_entropy_concentration(py::module& m);
void bind_cube2mat_n_gini(py::module& m);
void bind_cube2mat_ret_skew(py::module& m);
void bind_cube2mat_rv_entropy_concentration(py::module& m);
void bind_cube2mat_rv_gini_concentration(py::module& m);
void bind_cube2mat_trade_size_gini(py::module& m);
void bind_cube2mat_trend_resid_kurt(py::module& m);
void bind_cube2mat_trend_resid_skew(py::module& m);
void bind_cube2mat_volume_entropy_concentration(py::module& m);
void bind_cube2mat_volume_gini(py::module& m);
void bind_cube2mat_volume_median_to_mean_ratio(py::module& m);
void bind_cube2mat_volume_skew(py::module& m);
void bind_cube2mat_katz_fd_close(py::module& m);
void bind_cube2mat_oc_efficiency_over_range(py::module& m);
void bind_cube2mat_path_efficiency(py::module& m);
void bind_cube2mat_path_length_over_range(py::module& m);
void bind_cube2mat_total_variation_close(py::module& m);

PYBIND11_MODULE(machine_alpha, m) {
    m.doc() = "collection of intraday features";
    bind_alpha_example(m);
    bind_cube2mat_gini_absret(m);
    bind_cube2mat_n_entropy_concentration(m);
    bind_cube2mat_n_gini(m);
    bind_cube2mat_ret_skew(m);
    bind_cube2mat_rv_entropy_concentration(m);
    bind_cube2mat_rv_gini_concentration(m);
    bind_cube2mat_trade_size_gini(m);
    bind_cube2mat_trend_resid_kurt(m);
    bind_cube2mat_trend_resid_skew(m);
    bind_cube2mat_volume_entropy_concentration(m);
    bind_cube2mat_volume_gini(m);
    bind_cube2mat_volume_median_to_mean_ratio(m);
    bind_cube2mat_volume_skew(m);
    bind_cube2mat_katz_fd_close(m);
    bind_cube2mat_oc_efficiency_over_range(m);
    bind_cube2mat_path_efficiency(m);
    bind_cube2mat_path_length_over_range(m);
    bind_cube2mat_total_variation_close(m);
}
