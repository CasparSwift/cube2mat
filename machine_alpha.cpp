#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_cube2mat_vwap(py::module& m);
void bind_gini_absret(py::module& m);
void bind_n_entropy_concentration(py::module& m);
void bind_n_gini(py::module& m);
void bind_ret_skew(py::module& m);
void bind_rv_entropy_concentration(py::module& m);
void bind_rv_gini_concentration(py::module& m);
void bind_trade_size_gini(py::module& m);
void bind_trend_resid_kurt(py::module& m);
void bind_trend_resid_skew(py::module& m);
void bind_volume_entropy_concentration(py::module& m);
void bind_volume_gini(py::module& m);
void bind_volume_median_to_mean_ratio(py::module& m);
void bind_volume_skew(py::module& m);
void bind_katz_fd_close(py::module& m);
void bind_oc_efficiency_over_range(py::module& m);
void bind_path_efficiency(py::module& m);
void bind_path_length_over_range(py::module& m);
void bind_total_variation_close(py::module& m);
void bind_close_to_edge_distance_frac(py::module& m);
void bind_end_position_in_range(py::module& m);
void bind_intraday_max_drawdown_close(py::module& m);
void bind_parkinson_vol(py::module& m);
void bind_range_per_trade(py::module& m);
void bind_vwap_position_in_range(py::module& m);

PYBIND11_MODULE(machine_alpha, m) {
    m.doc() = "collection of intraday features";
    bind_cube2mat_vwap(m);
    bind_gini_absret(m);
    bind_n_entropy_concentration(m);
    bind_n_gini(m);
    bind_ret_skew(m);
    bind_rv_entropy_concentration(m);
    bind_rv_gini_concentration(m);
    bind_trade_size_gini(m);
    bind_trend_resid_kurt(m);
    bind_trend_resid_skew(m);
    bind_volume_entropy_concentration(m);
    bind_volume_gini(m);
    bind_volume_median_to_mean_ratio(m);
    bind_volume_skew(m);
    bind_katz_fd_close(m);
    bind_oc_efficiency_over_range(m);
    bind_path_efficiency(m);
    bind_path_length_over_range(m);
    bind_total_variation_close(m);
    bind_close_to_edge_distance_frac(m);
    bind_end_position_in_range(m);
    bind_intraday_max_drawdown_close(m);
    bind_parkinson_vol(m);
    bind_range_per_trade(m);
    bind_vwap_position_in_range(m);
}
