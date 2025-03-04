import pandas as pd

class mfRRPriceData:
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file, parse_dates=['HourDK'])
        self.data.sort_values('HourDK', inplace=True)
        self.data.reset_index(drop=True, inplace=True)


# #include <pybind11/pybind11.h>
# #include <pybind11/stl.h>
# #include <cmath>
# #include <vector>
# #include <tuple>
# #include <algorithm>
# #include <limits>

# namespace py = pybind11;

# // Computes the root-mean-square error of a profile relative to a target value.
# double rmse(const std::vector<double>& profile, double target) {
#     if (profile.empty()) {
#         return std::numeric_limits<double>::infinity();
#     }
#     double mse = 0.0;
#     for (auto p : profile) {
#         double diff = p - target;
#         mse += diff * diff;
#     }
#     mse /= profile.size();
#     return std::sqrt(mse);
# }

# // Computes the coefficient of variation of the profile.
# double cv(const std::vector<double>& profile) {
#     if (profile.empty()) {
#         return std::numeric_limits<double>::infinity();
#     }
#     double sum = 0.0;
#     for (auto p : profile) {
#         sum += p;
#     }
#     double mean = sum / profile.size();
#     if (mean == 0) return 0.0;
#     double variance = 0.0;
#     for (auto p : profile) {
#         double diff = p - mean;
#         variance += diff * diff;
#     }
#     variance /= profile.size();
#     return std::sqrt(variance) / mean;
# }

# // Performs a simple binary aggregation: it sums the two input profiles elementwise for the overlapping part.
# // Any remaining slices in the longer profile are appended.
# std::vector<double> binary_aggregate(const std::vector<double>& profile1,
#                                        const std::vector<double>& profile2) {
#     size_t len1 = profile1.size();
#     size_t len2 = profile2.size();
#     size_t min_len = std::min(len1, len2);
#     std::vector<double> aggregated;
#     aggregated.reserve(std::max(len1, len2));
#     for (size_t i = 0; i < min_len; i++) {
#         aggregated.push_back(profile1[i] + profile2[i]);
#     }
#     if (len1 > min_len) {
#         for (size_t i = min_len; i < len1; i++) {
#             aggregated.push_back(profile1[i]);
#         }
#     } else if (len2 > min_len) {
#         for (size_t i = min_len; i < len2; i++) {
#             aggregated.push_back(profile2[i]);
#         }
#     }
#     return aggregated;
# }



# // Checks if every slice in the profile is within target ± allowed_deviation.
# bool check_criteria(const std::vector<double>& profile, double target, double allowed_deviation) {
#     for (auto p : profile) {
#         if (p < target - allowed_deviation || p > target + allowed_deviation)
#             return false;
#     }
#     return true;
# }

# // A simple LP-style aggregation function.
# // Given a set of profiles (each represented as a vector of doubles), it selects the longest profile as a seed,
# // then incrementally aggregates other profiles (using binary_aggregate) if doing so reduces the RMSE with respect to target.
# // This simplified approach returns the aggregated profile.
# std::vector<double> aggregate_profiles(const std::vector<std::vector<double>>& profiles,
#                                          double target, double allowed_deviation) {
#     if (profiles.empty()) {
#         return {};
#     }
#     // Copy profiles into a modifiable list.
#     std::vector<std::vector<double>> remaining = profiles;

#     // Select seed: choose the profile with the maximum length.
#     auto seed_it = std::max_element(remaining.begin(), remaining.end(),
#         [](const std::vector<double>& a, const std::vector<double>& b) {
#             return a.size() < b.size();
#         });
#     std::vector<double> current = *seed_it;
#     remaining.erase(seed_it);

#     double current_rmse = rmse(current, target);
#     bool improved = true;
#     while (improved && !remaining.empty()) {
#         improved = false;
#         int best_index = -1;
#         std::vector<double> best_new_profile;
#         double best_error = current_rmse;
#         // Try aggregating each remaining profile with the current aggregation.
#         for (size_t i = 0; i < remaining.size(); i++) {
#             const auto& candidate = remaining[i];
#             std::vector<double> new_profile = binary_aggregate(current, candidate);
#             double new_error = rmse(new_profile, target);
#             // Choose candidate if error improves.
#             if (new_error < best_error) {
#                 best_error = new_error;
#                 best_new_profile = new_profile;
#                 best_index = static_cast<int>(i);
#             }
#         }
#         if (best_index != -1) {
#             current = best_new_profile;
#             current_rmse = best_error;
#             remaining.erase(remaining.begin() + best_index);
#             improved = true;
#         }
#     }
#     return current;
# }

# // Convenience function that aggregates profiles and returns a Python dict with extra info.
# py::dict aggregate_and_check(const std::vector<std::vector<double>>& profiles,
#                              double target, double allowed_deviation) {
#     std::vector<double> agg_profile = aggregate_profiles(profiles, target, allowed_deviation);
#     bool meets = check_criteria(agg_profile, target, allowed_deviation);
#     py::dict result;
#     result["aggregated_profile"] = agg_profile;
#     result["meets_criteria"] = meets;
#     result["rmse"] = rmse(agg_profile, target);
#     result["cv"] = cv(agg_profile);
#     return result;
# }

# PYBIND11_MODULE(aggregation, m) {
#     m.doc() = "Aggregation functions library for EV flex-offer aggregation using LP method";
#     m.def("rmse", &rmse, "Compute RMSE of a profile against a target value",
#           py::arg("profile"), py::arg("target"));
#     m.def("cv", &cv, "Compute the coefficient of variation of a profile",
#           py::arg("profile"));
#     m.def("binary_aggregate", &binary_aggregate, "Aggregate two profiles elementwise",
#           py::arg("profile1"), py::arg("profile2"));
#     m.def("check_criteria", &check_criteria,
#           "Check if a profile meets the criteria (target ± allowed deviation)",
#           py::arg("profile"), py::arg("target"), py::arg("allowed_deviation"));
#     m.def("aggregate_profiles", &aggregate_profiles,
#           "Aggregate a set of profiles to approach the target value",
#           py::arg("profiles"), py::arg("target"), py::arg("allowed_deviation"));
#     m.def("aggregate_and_check", &aggregate_and_check,
#           "Aggregate profiles and return a dict with the aggregated profile and statistics",
#           py::arg("profiles"), py::arg("target"), py::arg("allowed_deviation"));
# }
