#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <pybind11/numpy.h>


#include "theia/matching/feature_matcher.h"
#include "theia/matching/feature_matcher_options.h"
#include "theia/matching/features_and_matches_database.h"
#include "theia/sfm/two_view_match_geometric_verification.h"
#include "theia/matching/brute_force_feature_matcher.h"
#include "theia/matching/cascade_hashing_feature_matcher.h"
#include "theia/matching/cascade_hasher.h"

namespace py = pybind11;


PYBIND11_MODULE(pytheia_matching, m) {

    //FeaturesAndMatchesDatabase
    py::class_<theia::FeaturesAndMatchesDatabase>(m, "FeaturesAndMatchesDatabase")
    ;


    //TwoViewMatchGeometricVerification Options
    py::class_<theia::TwoViewMatchGeometricVerification::Options>(m, "Options")

      .def_readwrite("num_threads", &theia::TwoViewMatchGeometricVerification::Options::min_num_inlier_matches)
      .def_readwrite("keep_only_symmetric_matches", &theia::TwoViewMatchGeometricVerification::Options::guided_matching)
      .def_readwrite("use_lowes_ratio", &theia::TwoViewMatchGeometricVerification::Options::guided_matching_max_distance_pixels)
      .def_readwrite("lowes_ratio", &theia::TwoViewMatchGeometricVerification::Options::guided_matching_lowes_ratio)
      .def_readwrite("perform_geometric_verification", &theia::TwoViewMatchGeometricVerification::Options::bundle_adjustment)
      .def_readwrite("min_num_feature_matches", &theia::TwoViewMatchGeometricVerification::Options::triangulation_max_reprojection_error)
      .def_readwrite("min_triangulation_angle_degrees", &theia::TwoViewMatchGeometricVerification::Options::min_triangulation_angle_degrees)
      .def_readwrite("final_max_reprojection_error", &theia::TwoViewMatchGeometricVerification::Options::final_max_reprojection_error)

    ;

    //FeatureMatcherOptions
    py::class_<theia::FeatureMatcherOptions>(m, "FeatureMatcherOptions")

      .def_readwrite("num_threads", &theia::FeatureMatcherOptions::num_threads)
      .def_readwrite("keep_only_symmetric_matches", &theia::FeatureMatcherOptions::keep_only_symmetric_matches)
      .def_readwrite("use_lowes_ratio", &theia::FeatureMatcherOptions::use_lowes_ratio)
      .def_readwrite("lowes_ratio", &theia::FeatureMatcherOptions::lowes_ratio)
      .def_readwrite("perform_geometric_verification", &theia::FeatureMatcherOptions::perform_geometric_verification)
      .def_readwrite("min_num_feature_matches", &theia::FeatureMatcherOptions::min_num_feature_matches)
      .def_readwrite("geometric_verification_options", &theia::FeatureMatcherOptions::geometric_verification_options)

    ;



    //FeatureMatcher
    py::class_<theia::FeatureMatcher>(m, "FeatureMatcher")
      //abstract class in the constructor
      //.def(py::init<theia::FeatureMatcherOptions, &theia::FeaturesAndMatchesDatabase>())
      //.def("AddImages", &theia::FeatureMatcher::AddImages)
      //.def("AddImages", &theia::FeatureMatcher::AddImages)
      .def("AddImages", (void (theia::FeatureMatcher::*)(const std::vector<std::string>& )) &theia::FeatureMatcher::AddImages, py::return_value_policy::reference_internal)
      .def("AddImages", (void (theia::FeatureMatcher::*)(const std::string& )) &theia::FeatureMatcher::AddImages, py::return_value_policy::reference_internal)
      .def("MatchImages", &theia::FeatureMatcher::MatchImages)
      .def("SetImagePairsToMatch", &theia::FeatureMatcher::SetImagePairsToMatch)

    ;

    //BruteForceFeatureMatcher
    py::class_<theia::BruteForceFeatureMatcher, theia::FeatureMatcher>(m, "BruteForceFeatureMatcher")
      //abstract class in the constructor
      //.def(py::init<theia::FeatureMatcherOptions, theia::FeaturesAndMatchesDatabase>())


    ;

    //CascadeHashingFeatureMatcher
    py::class_<theia::CascadeHashingFeatureMatcher, theia::FeatureMatcher>(m, "CascadeHashingFeatureMatcher")
      //abstract class in the constructor
      //.def(py::init<theia::FeatureMatcherOptions, theia::FeaturesAndMatchesDatabase>())
      .def("AddImages", (void (theia::CascadeHashingFeatureMatcher::*)(const std::vector<std::string>& )) &theia::CascadeHashingFeatureMatcher::AddImages, py::return_value_policy::reference_internal)
      .def("AddImages", (void (theia::CascadeHashingFeatureMatcher::*)(const std::string& )) &theia::CascadeHashingFeatureMatcher::AddImages, py::return_value_policy::reference_internal)


    ;


}
