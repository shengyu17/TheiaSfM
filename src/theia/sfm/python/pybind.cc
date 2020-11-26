#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include <glog/logging.h>
#include <math.h>
#include <Eigen/Dense>
#include <complex>
#include <algorithm>

#include "theia/math/polynomial.h"
#include "theia/sfm/pose/util.h"
//#include "include/theia/theia.h"

// header files
#include "theia/sfm/pose/perspective_three_point.h"
#include "theia/sfm/pose/eight_point_fundamental_matrix.h"
#include "theia/sfm/pose/five_point_relative_pose.h"
#include "theia/sfm/pose/five_point_focal_length_radial_distortion.h"
#include "theia/sfm/pose/four_point_focal_length.h"
#include "theia/sfm/pose/four_point_focal_length_radial_distortion.h"
#include "theia/sfm/pose/four_point_homography.h"
#include "theia/sfm/pose/four_point_relative_pose_partial_rotation.h"
#include "theia/sfm/pose/three_point_relative_pose_partial_rotation.h"
#include "theia/sfm/pose/two_point_pose_partial_rotation.h"
#include "theia/sfm/triangulation/triangulation.h"
#include "theia/sfm/pose/dls_pnp.h"
#include "theia/sfm/pose/position_from_two_rays.h"
#include "theia/sfm/pose/relative_pose_from_two_points_with_known_rotation.h"
#include "theia/sfm/pose/seven_point_fundamental_matrix.h"
#include "theia/sfm/pose/sim_transform_partial_rotation.h"
#include "theia/sfm/pose/essential_matrix_utils.h"
#include "theia/matching/feature_correspondence.h"
#include "theia/sfm/pose/fundamental_matrix_util.h"

namespace py = pybind11;

#include <vector>
#include <iostream>
#include <pybind11/numpy.h>

PYBIND11_MODULE(pytheia_sfm, m) {

  py::class_<theia::FeatureCorrespondence>(m, "FeatureCorrespondence")
    .def(py::init())
    .def(py::init<Eigen::Vector2d, Eigen::Vector2d>())
    .def_readwrite("feature1", &theia::FeatureCorrespondence::feature1)
    .def_readwrite("feature2", &theia::FeatureCorrespondence::feature2)
  ;


  //pose
  m.def("PoseFromThreePoints", theia::PoseFromThreePointsWrapper);
  m.def("NormalizedEightPointFundamentalMatrix", theia::NormalizedEightPointFundamentalMatrixWrapper);
  m.def("FivePointRelativePose", theia::FivePointRelativePoseWrapper);
  m.def("FourPointPoseAndFocalLength", theia::FourPointPoseAndFocalLengthWrapper);
  m.def("FourPointHomography", theia::FourPointHomographyWrapper);
  m.def("FourPointsPoseFocalLengthRadialDistortion", theia::FourPointsPoseFocalLengthRadialDistortionWrapper);
  m.def("FourPointRelativePosePartialRotation", theia::FourPointRelativePosePartialRotationWrapper);
  //m.def("FivePointFocalLengthRadialDistortion", theia::FivePointFocalLengthRadialDistortionWrapper);
  m.def("ThreePointRelativePosePartialRotation", theia::ThreePointRelativePosePartialRotationWrapper);
  m.def("TwoPointPosePartialRotation", theia::TwoPointPosePartialRotationWrapper);
  m.def("DlsPnp", theia::DlsPnpWrapper);
  m.def("PositionFromTwoRays", theia::PositionFromTwoRaysWrapper);
  m.def("RelativePoseFromTwoPointsWithKnownRotation", theia::RelativePoseFromTwoPointsWithKnownRotationWrapper);
  m.def("SevenPointFundamentalMatrix", theia::SevenPointFundamentalMatrixWrapper);
  m.def("SimTransformPartialRotation", theia::SimTransformPartialRotationWrapper);
  m.def("DecomposeEssentialMatrix", theia::DecomposeEssentialMatrixWrapper);
  m.def("EssentialMatrixFromTwoProjectionMatrices", theia::EssentialMatrixFromTwoProjectionMatricesWrapper);
  m.def("GetBestPoseFromEssentialMatrix", theia::GetBestPoseFromEssentialMatrixWrapper);
  m.def("FocalLengthsFromFundamentalMatrix", theia::FocalLengthsFromFundamentalMatrixWrapper);
  m.def("SharedFocalLengthsFromFundamentalMatrix", theia::SharedFocalLengthsFromFundamentalMatrixWrapper);
  m.def("ProjectionMatricesFromFundamentalMatrix", theia::ProjectionMatricesFromFundamentalMatrixWrapper);
  m.def("FundamentalMatrixFromProjectionMatrices", theia::FundamentalMatrixFromProjectionMatricesWrapper);
  m.def("EssentialMatrixFromFundamentalMatrix", theia::EssentialMatrixFromFundamentalMatrixWrapper);
  m.def("ComposeFundamentalMatrix", theia::ComposeFundamentalMatrixWrapper);




  //triangulation
  m.def("Triangulate", theia::TriangulateWrapper);
  m.def("TriangulateMidpoint", theia::TriangulateMidpointWrapper);
  m.def("TriangulateDLT", theia::TriangulateDLTWrapper);
  m.def("TriangulateNViewSVD", theia::TriangulateNViewSVDWrapper);
  m.def("TriangulateNView", theia::TriangulateNViewWrapper);
  //m.def("IsTriangulatedPointInFrontOfCameras", theia::IsTriangulatedPointInFrontOfCameras);
  m.def("SufficientTriangulationAngle", theia::SufficientTriangulationAngle);


}
