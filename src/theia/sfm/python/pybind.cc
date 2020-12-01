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


// header files
/*
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

#include "theia/sfm/pose/dls_pnp.h"
#include "theia/sfm/pose/position_from_two_rays.h"
#include "theia/sfm/pose/relative_pose_from_two_points_with_known_rotation.h"
#include "theia/sfm/pose/seven_point_fundamental_matrix.h"
#include "theia/sfm/pose/sim_transform_partial_rotation.h"
#include "theia/sfm/pose/essential_matrix_utils.h"
#include "theia/matching/feature_correspondence.h"
#include "theia/sfm/pose/fundamental_matrix_util.h"
*/
#include "theia/sfm/pose/pose_wrapper.h"
#include "theia/sfm/triangulation/triangulation.h"
#include "theia/sfm/triangulation/triangulation_wrapper.h"

#include "theia/sfm/transformation/transformation_wrapper.h"
#include "theia/sfm/transformation/align_point_clouds.h"
#include "theia/sfm/transformation/gdls_similarity_transform.h"

#include "theia/sfm/camera/camera.h"
#include "theia/sfm/camera/camera_wrapper.h"
#include "theia/sfm/camera/camera_intrinsics_model.h"
#include "theia/sfm/camera/division_undistortion_camera_model.h"
#include "theia/sfm/camera/fisheye_camera_model.h"
#include "theia/sfm/camera/fov_camera_model.h"
#include "theia/sfm/camera/pinhole_camera_model.h"
#include "theia/sfm/camera/pinhole_radial_tangential_camera_model.h"


namespace py = pybind11;
#include <vector>
#include <iostream>
#include <pybind11/numpy.h>

template <int N>
void AddIntrinsicsPriorType(py::module& m, const std::string& name) {
  py::class_<theia::Prior<N>>(m, ("Prior" + name).c_str())
      .def(py::init())
      .def_readwrite("is_set", &theia::Prior<N>::is_set)
      //.def_readwrite("value", &theia::Prior<N>::value)
    ;
  }

PYBIND11_MODULE(pytheia_sfm, m) {
  //matching
  py::class_<theia::FeatureCorrespondence>(m, "FeatureCorrespondence")
    .def(py::init())
    .def(py::init<Eigen::Vector2d, Eigen::Vector2d>())
    .def_readwrite("feature1", &theia::FeatureCorrespondence::feature1)
    .def_readwrite("feature2", &theia::FeatureCorrespondence::feature2)
  ;

  //camera
  AddIntrinsicsPriorType<1>(m, "Scalar");
  AddIntrinsicsPriorType<2>(m, "Vector2d");
  AddIntrinsicsPriorType<3>(m, "Vector3d");
  AddIntrinsicsPriorType<4>(m, "Vector4d");
  /*
  AddIntrinsicsPriorType<1>(m, "Focallength");
  AddIntrinsicsPriorType<2>(m, "Principalpoint");
  AddIntrinsicsPriorType<1>(m, "Aspectratio");
  AddIntrinsicsPriorType<1>(m, "Skew");
  AddIntrinsicsPriorType<4>(m, "Radialdistortion");
  AddIntrinsicsPriorType<2>(m, "Tangentialdistortion");
  AddIntrinsicsPriorType<3>(m, "Position");
  AddIntrinsicsPriorType<3>(m, "Orientation");
  AddIntrinsicsPriorType<1>(m, "Latitude");
  AddIntrinsicsPriorType<1>(m, "Longitude");
  AddIntrinsicsPriorType<1>(m, "Altitude");
  */
  py::class_<theia::CameraIntrinsicsModel>(m, "CameraIntrinsicsModel")


  ;
  py::class_<theia::FisheyeCameraModel, theia::CameraIntrinsicsModel>(m, "FisheyeCameraModel")
    .def(py::init<>())
    .def("Type", &theia::FisheyeCameraModel::Type)
    .def("NumParameters", &theia::FisheyeCameraModel::NumParameters)
    .def("SetFromCameraIntrinsicsPriors", &theia::FisheyeCameraModel::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::FisheyeCameraModel::CameraIntrinsicsPriorFromIntrinsics)
    .def("GetSubsetFromOptimizeIntrinsicsType", &theia::FisheyeCameraModel::GetSubsetFromOptimizeIntrinsicsType)
    .def("GetCalibrationMatrix", &theia::FisheyeCameraModel::GetCalibrationMatrix)
    .def("PrintIntrinsics", &theia::FisheyeCameraModel::PrintIntrinsics)
  ;
  m.def("ComposeProjectionMatrix", theia::ComposeProjectionMatrixWrapper);
  m.def("DecomposeProjectionMatrix", theia::DecomposeProjectionMatrixWrapper);
  m.def("CalibrationMatrixToIntrinsics", theia::CalibrationMatrixToIntrinsicsWrapper);
  m.def("IntrinsicsToCalibrationMatrix", theia::IntrinsicsToCalibrationMatrixWrapper);

//  py::class_<theia::Prior>(m, "Prior")
//    .def(py::init())
//    .def(py::init<Eigen::Vector2d, Eigen::Vector2d>())
//    .def_readwrite("is_set", &theia::Prior::is_set)
//    .def_readwrite("value", &theia::Prior::value)
//  ;

  py::class_<theia::Camera>(m, "Camera")
    .def(py::init())
    .def(py::init<theia::Camera>())
    .def("ImageWidth", &theia::Camera::ImageWidth)
    .def("ImageHeight", &theia::Camera::ImageHeight)
    //.def_readonly_static("kExtrinsicsSize", &theia::Camera::kExtrinsicsSize)
  ;


  // tested
  py::enum_<theia::CameraIntrinsicsModelType>(m, "CameraIntrinsicsModelType")
    .value("INVALID", theia::CameraIntrinsicsModelType::INVALID)
    .value("PINHOLE", theia::CameraIntrinsicsModelType::PINHOLE)
    .value("PINHOLE_RADIAL_TANGENTIAL", theia::CameraIntrinsicsModelType::PINHOLE_RADIAL_TANGENTIAL)
    .value("FISHEYE", theia::CameraIntrinsicsModelType::FISHEYE)
    .value("FOV", theia::CameraIntrinsicsModelType::FOV)
    .value("DIVISION_UNDISTORTION", theia::CameraIntrinsicsModelType::DIVISION_UNDISTORTION)
    .export_values()
  ;

  py::class_<theia::CameraIntrinsicsPrior>(m, "CameraIntrinsicsPrior")
    .def(py::init())

    .def_readwrite("image_width", &theia::CameraIntrinsicsPrior::image_width)
    .def_readwrite("image_height", &theia::CameraIntrinsicsPrior::image_height)
    .def_readwrite("camera_intrinsics_model_type", &theia::CameraIntrinsicsPrior::camera_intrinsics_model_type)
    .def_readwrite("focal_length", &theia::CameraIntrinsicsPrior::focal_length)
    .def_readwrite("principal_point", &theia::CameraIntrinsicsPrior::principal_point)
    .def_readwrite("aspect_ratio", &theia::CameraIntrinsicsPrior::aspect_ratio)
    .def_readwrite("skew", &theia::CameraIntrinsicsPrior::skew)
    .def_readwrite("radial_distortion", &theia::CameraIntrinsicsPrior::radial_distortion)
    .def_readwrite("tangential_distortion", &theia::CameraIntrinsicsPrior::tangential_distortion)
    .def_readwrite("position", &theia::CameraIntrinsicsPrior::position)
    .def_readwrite("orientation", &theia::CameraIntrinsicsPrior::orientation)
    .def_readwrite("latitude", &theia::CameraIntrinsicsPrior::latitude)
    .def_readwrite("longitude", &theia::CameraIntrinsicsPrior::longitude)
    .def_readwrite("altitude", &theia::CameraIntrinsicsPrior::altitude)

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

  //transformation
  m.def("AlignPointCloudsUmeyama", theia::AlignPointCloudsUmeyamaWrapper);
  m.def("AlignPointCloudsUmeyamaWithWeights", theia::AlignPointCloudsUmeyamaWithWeightsWrapper);
  m.def("GdlsSimilarityTransform" ,theia::GdlsSimilarityTransformWrapper);
  m.def("AlignRotations", theia::AlignRotationsWrapper);



  //triangulation
  m.def("Triangulate", theia::TriangulateWrapper);
  m.def("TriangulateMidpoint", theia::TriangulateMidpointWrapper);
  m.def("TriangulateDLT", theia::TriangulateDLTWrapper);
  m.def("TriangulateNViewSVD", theia::TriangulateNViewSVDWrapper);
  m.def("TriangulateNView", theia::TriangulateNViewWrapper);
  m.def("IsTriangulatedPointInFrontOfCameras", theia::IsTriangulatedPointInFrontOfCameras);
  m.def("SufficientTriangulationAngle", theia::SufficientTriangulationAngle);


}
