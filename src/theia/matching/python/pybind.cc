#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <pybind11/numpy.h>

#include "theia/matching/keypoints_and_descriptors.h"
#include "theia/matching/indexed_feature_match.h"
#include "theia/matching/feature_matcher.h"
#include "theia/matching/feature_matcher_options.h"
#include "theia/matching/features_and_matches_database.h"
#include "theia/sfm/two_view_match_geometric_verification.h"
#include "theia/matching/brute_force_feature_matcher.h"
#include "theia/matching/cascade_hashing_feature_matcher.h"
#include "theia/matching/cascade_hasher.h"
#include "theia/matching/rocksdb_features_and_matches_database.h"

namespace py = pybind11;


PYBIND11_MODULE(pytheia_matching, m) {

    //FeaturesAndMatchesDatabase
    py::class_<theia::FeaturesAndMatchesDatabase /*, theia::PyFeaturesAndMatchesDatabase */>(m, "FeaturesAndMatchesDatabase")
      //.def(py::init<>())
      //.def("ContainsCameraIntrinsicsPrior", &theia::FeaturesAndMatchesDatabase::ContainsCameraIntrinsicsPrior)
    ;

    //RocksDbFeaturesAndMatchesDatabase

    py::class_<theia::RocksDbFeaturesAndMatchesDatabase, theia::FeaturesAndMatchesDatabase>(m, "RocksDbFeaturesAndMatchesDatabase")
      .def(py::init<std::string>())
      .def("ContainsCameraIntrinsicsPrior", &theia::RocksDbFeaturesAndMatchesDatabase::ContainsCameraIntrinsicsPrior)
    ;


    //TwoViewMatchGeometricVerification Options
    py::class_<theia::TwoViewMatchGeometricVerification::Options>(m, "TwoViewMatchGeometricVerificationOptions")

      .def_readwrite("min_num_inlier_matches", &theia::TwoViewMatchGeometricVerification::Options::min_num_inlier_matches)
      .def_readwrite("guided_matching", &theia::TwoViewMatchGeometricVerification::Options::guided_matching)
      .def_readwrite("guided_matching_max_distance_pixels", &theia::TwoViewMatchGeometricVerification::Options::guided_matching_max_distance_pixels)
      .def_readwrite("guided_matching_lowes_ratio", &theia::TwoViewMatchGeometricVerification::Options::guided_matching_lowes_ratio)
      .def_readwrite("bundle_adjustment", &theia::TwoViewMatchGeometricVerification::Options::bundle_adjustment)
      .def_readwrite("triangulation_max_reprojection_error", &theia::TwoViewMatchGeometricVerification::Options::triangulation_max_reprojection_error)
      .def_readwrite("min_triangulation_angle_degrees", &theia::TwoViewMatchGeometricVerification::Options::min_triangulation_angle_degrees)
      .def_readwrite("final_max_reprojection_error", &theia::TwoViewMatchGeometricVerification::Options::final_max_reprojection_error)

    ;

    //FeatureMatcherOptions
    py::class_<theia::FeatureMatcherOptions>(m, "FeatureMatcherOptions")
      .def(py::init<>())
      .def_readwrite("num_threads", &theia::FeatureMatcherOptions::num_threads)
      .def_readwrite("keep_only_symmetric_matches", &theia::FeatureMatcherOptions::keep_only_symmetric_matches)
      .def_readwrite("use_lowes_ratio", &theia::FeatureMatcherOptions::use_lowes_ratio)
      .def_readwrite("lowes_ratio", &theia::FeatureMatcherOptions::lowes_ratio)
      .def_readwrite("perform_geometric_verification", &theia::FeatureMatcherOptions::perform_geometric_verification)
      .def_readwrite("min_num_feature_matches", &theia::FeatureMatcherOptions::min_num_feature_matches)
      .def_readwrite("geometric_verification_options", &theia::FeatureMatcherOptions::geometric_verification_options)

    ;

    //KeypointsAndDescriptors
    py::class_<theia::KeypointsAndDescriptors>(m, "KeypointsAndDescriptors")
      .def(py::init<>())
      .def_readwrite("image_name", &theia::KeypointsAndDescriptors::image_name)
      .def_readwrite("keypoints", &theia::KeypointsAndDescriptors::keypoints)
      .def_readwrite("descriptors", &theia::KeypointsAndDescriptors::descriptors)
    ;

    //IndexedFeatureMatch
    py::class_<theia::IndexedFeatureMatch>(m, "IndexedFeatureMatch")
      .def(py::init<>())
      .def(py::init<int, int, float>())
      .def_readwrite("feature1_ind", &theia::IndexedFeatureMatch::feature1_ind)
      .def_readwrite("feature2_ind", &theia::IndexedFeatureMatch::feature2_ind)
      .def_readwrite("distance", &theia::IndexedFeatureMatch::distance)
    ;



    //FeatureMatcher
    py::class_<theia::FeatureMatcher>(m, "FeatureMatcher")
      //abstract class in the constructor
      //.def(py::init<theia::FeatureMatcherOptions, &theia::FeaturesAndMatchesDatabase>())
      .def("AddImages", (void (theia::FeatureMatcher::*)(const std::vector<std::string>& )) &theia::FeatureMatcher::AddImages, py::return_value_policy::reference_internal)
      .def("AddImages", (void (theia::FeatureMatcher::*)(const std::string& )) &theia::FeatureMatcher::AddImages, py::return_value_policy::reference_internal)
      .def("MatchImages", &theia::FeatureMatcher::MatchImages)
      .def("SetImagePairsToMatch", &theia::FeatureMatcher::SetImagePairsToMatch)

    ;

    //BruteForceFeatureMatcher
    py::class_<theia::BruteForceFeatureMatcher, theia::FeatureMatcher>(m, "BruteForceFeatureMatcher")
      .def(py::init<theia::FeatureMatcherOptions, theia::FeaturesAndMatchesDatabase*>())
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
