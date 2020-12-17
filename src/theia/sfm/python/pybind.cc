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
#include "theia/sfm/transformation/align_reconstructions.h"

#include "theia/sfm/camera/camera.h"
#include "theia/sfm/camera/camera_wrapper.h"
#include "theia/sfm/camera/camera_intrinsics_model.h"
#include "theia/sfm/camera/division_undistortion_camera_model.h"
#include "theia/sfm/camera/fisheye_camera_model.h"
#include "theia/sfm/camera/fov_camera_model.h"
#include "theia/sfm/camera/pinhole_camera_model.h"
#include "theia/sfm/camera/pinhole_radial_tangential_camera_model.h"

#include "theia/sfm/view.h"
#include "theia/sfm/track.h"
#include "theia/sfm/reconstruction.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/view_graph/view_graph.h"
#include "theia/sfm/gps_converter.h"

#include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
#include "theia/sfm/bundle_adjustment/create_loss_function.h"
#include "theia/sfm/bundle_adjustment/bundle_adjustment_wrapper.h"
#include "theia/sfm/bundle_adjustment/bundle_adjuster.h"
#include "theia/sfm/bundle_adjustment/bundle_adjust_two_views.h"
#include "theia/sfm/bundle_adjustment/optimize_relative_position_with_known_rotation.h"


#include "theia/sfm/global_pose_estimation/position_estimator.h"
#include "theia/sfm/global_pose_estimation/rotation_estimator.h"
#include "theia/sfm/global_pose_estimation/robust_rotation_estimator.h"
#include "theia/sfm/global_pose_estimation/least_unsquared_deviation_position_estimator.h"
#include "theia/sfm/global_pose_estimation/linear_position_estimator.h"
#include "theia/sfm/global_pose_estimation/linear_rotation_estimator.h"
#include "theia/sfm/global_pose_estimation/nonlinear_position_estimator.h"
#include "theia/sfm/global_pose_estimation/nonlinear_rotation_estimator.h"
#include "theia/sfm/global_pose_estimation/global_pose_estimation_wrapper.h"

// for overloaded function in CameraInstrinsicsModel
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

namespace py = pybind11;
#include <vector>
#include <iostream>
#include <pybind11/numpy.h>

// Initialize gtest
//google::InitGoogleLogging(argv[0]);

template <int N>
void AddIntrinsicsPriorType(py::module& m, const std::string& name) {
  py::class_<theia::Prior<N>>(m, ("Prior" + name).c_str())
      .def(py::init())
      .def_readwrite("is_set", &theia::Prior<N>::is_set)
      .def_property("value", &theia::Prior<N>::GetParametersValues, &theia::Prior<N>::SetParametersValues)
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

  // abstract superclass and 5 subclasses (camera models)
  py::class_<theia::CameraIntrinsicsModel, std::shared_ptr<theia::CameraIntrinsicsModel>> camera_intrinsics_model(m, "CameraIntrinsicsModel");
  camera_intrinsics_model.def_property("FocalLength", &theia::CameraIntrinsicsModel::FocalLength, &theia::CameraIntrinsicsModel::SetFocalLength)
    .def_property_readonly("PrincipalPointX", &theia::CameraIntrinsicsModel::PrincipalPointX)
    .def_property_readonly("PrincipalPointY", &theia::CameraIntrinsicsModel::PrincipalPointY)
    .def("SetPrincipalPoint", &theia::CameraIntrinsicsModel::SetPrincipalPoint)
    .def("CameraToImageCoordinates", &theia::CameraIntrinsicsModel::CameraToImageCoordinates)
    .def("ImageToCameraCoordinates", &theia::CameraIntrinsicsModel::ImageToCameraCoordinates)
    .def("GetParameter", &theia::CameraIntrinsicsModel::GetParameter)
    .def("SetParameter", &theia::CameraIntrinsicsModel::SetParameter)

    //.def("DistortPoint", py::overload_cast<const Eigen::Vector2d>(&theia::CameraIntrinsicsModel::DistortPoint, py::const_))
    //.def("DistortPoint",   py::overload_cast<const Eigen::Vector2d&>(&theia::CameraIntrinsicsModel::DistortPoint, py::const_));
    //.def("DistortPoint", static_cast<Eigen::Vector2d (theia::CameraIntrinsicsModel::*)(const Eigen::Vector2d &)>(&theia::CameraIntrinsicsModel::DistortPoint), "Set the pet's name")
    //.def("DistortPoint", overload_cast_<const Eigen::Vector2d>()(&theia::CameraIntrinsicsModel::DistortPoint), "Set the pet's age")


    .def("DistortPoint",
         (Eigen::Vector2d
          (theia::CameraIntrinsicsModel::*)(const Eigen::Vector2d& )) &theia::CameraIntrinsicsModel::DistortPoint,
         py::return_value_policy::reference_internal)

    //.def("UndistortPoint",(Eigen::Vector2d (theia::CameraIntrinsicsModel::*)(const Eigen::Vector2d& )) &
    //     theia::CameraIntrinsicsModel::UndistortPoint, py::arg("UndistortPoint"))





  ;
  // FisheyeCameraModel
  py::class_<theia::FisheyeCameraModel, std::shared_ptr<theia::FisheyeCameraModel>>(m, "FisheyeCameraModel", camera_intrinsics_model)
    .def(py::init<>())
    .def("Type", &theia::FisheyeCameraModel::Type)
    .def("NumParameters", &theia::FisheyeCameraModel::NumParameters)
    .def("SetFromCameraIntrinsicsPriors", &theia::FisheyeCameraModel::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::FisheyeCameraModel::CameraIntrinsicsPriorFromIntrinsics)
    // OptimizeIntrinsicsType not defined
    .def("GetSubsetFromOptimizeIntrinsicsType", &theia::FisheyeCameraModel::GetSubsetFromOptimizeIntrinsicsType)
    .def("GetCalibrationMatrix", &theia::FisheyeCameraModel::GetCalibrationMatrix)
    .def("PrintIntrinsics", &theia::FisheyeCameraModel::PrintIntrinsics)
    .def_property_readonly("kIntrinsicsSize", &theia::FisheyeCameraModel::NumParameters)
    .def_property("AspectRatio", &theia::FisheyeCameraModel::AspectRatio, &theia::FisheyeCameraModel::SetAspectRatio)
    .def_property("Skew", &theia::FisheyeCameraModel::Skew, &theia::FisheyeCameraModel::SetSkew)
    .def_property_readonly("RadialDistortion1", &theia::FisheyeCameraModel::RadialDistortion1)
    .def_property_readonly("RadialDistortion2", &theia::FisheyeCameraModel::RadialDistortion2)
    .def_property_readonly("RadialDistortion3", &theia::FisheyeCameraModel::RadialDistortion3)
    .def_property_readonly("RadialDistortion4", &theia::FisheyeCameraModel::RadialDistortion4)
    .def("SetRadialDistortion", &theia::FisheyeCameraModel::SetRadialDistortion)

  ;
  // PinholeRadialTangentialCameraModel
  py::class_<theia::PinholeRadialTangentialCameraModel, std::shared_ptr<theia::PinholeRadialTangentialCameraModel>>(m, "PinholeRadialTangentialCameraModel", camera_intrinsics_model)
    .def(py::init<>())
    .def("Type", &theia::PinholeRadialTangentialCameraModel::Type)
    .def("NumParameters", &theia::PinholeRadialTangentialCameraModel::NumParameters)
    .def("SetFromCameraIntrinsicsPriors", &theia::PinholeRadialTangentialCameraModel::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::PinholeRadialTangentialCameraModel::CameraIntrinsicsPriorFromIntrinsics)
    // OptimizeIntrinsicsType not defined
    .def("GetSubsetFromOptimizeIntrinsicsType", &theia::PinholeRadialTangentialCameraModel::GetSubsetFromOptimizeIntrinsicsType)
    .def("GetCalibrationMatrix", &theia::PinholeRadialTangentialCameraModel::GetCalibrationMatrix)
    .def("PrintIntrinsics", &theia::PinholeRadialTangentialCameraModel::PrintIntrinsics)
    .def_property_readonly("kIntrinsicsSize", &theia::PinholeRadialTangentialCameraModel::NumParameters)
    .def_property("AspectRatio", &theia::PinholeRadialTangentialCameraModel::AspectRatio, &theia::PinholeRadialTangentialCameraModel::SetAspectRatio)
    .def_property("Skew", &theia::PinholeRadialTangentialCameraModel::Skew, &theia::PinholeRadialTangentialCameraModel::SetSkew)
    .def_property_readonly("RadialDistortion1", &theia::PinholeRadialTangentialCameraModel::RadialDistortion1)
    .def_property_readonly("RadialDistortion2", &theia::PinholeRadialTangentialCameraModel::RadialDistortion2)
    .def_property_readonly("RadialDistortion3", &theia::PinholeRadialTangentialCameraModel::RadialDistortion3)
    .def_property_readonly("TangentialDistortion1", &theia::PinholeRadialTangentialCameraModel::TangentialDistortion1)
    .def_property_readonly("TangentialDistortion2", &theia::PinholeRadialTangentialCameraModel::TangentialDistortion2)
    .def("SetRadialDistortion", &theia::PinholeRadialTangentialCameraModel::SetRadialDistortion)
    .def("SetTangentialDistortion", &theia::PinholeRadialTangentialCameraModel::SetTangentialDistortion)

  ;
  // DivisionUndistortionCameraModel
  py::class_<theia::DivisionUndistortionCameraModel, std::shared_ptr<theia::DivisionUndistortionCameraModel>>(m, "DivisionUndistortionCameraModel", camera_intrinsics_model)
    .def(py::init<>())
    .def("Type", &theia::DivisionUndistortionCameraModel::Type)
    .def("NumParameters", &theia::DivisionUndistortionCameraModel::NumParameters)
    .def("SetFromCameraIntrinsicsPriors", &theia::DivisionUndistortionCameraModel::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::DivisionUndistortionCameraModel::CameraIntrinsicsPriorFromIntrinsics)
    // OptimizeIntrinsicsType not defined
    .def("GetSubsetFromOptimizeIntrinsicsType", &theia::DivisionUndistortionCameraModel::GetSubsetFromOptimizeIntrinsicsType)
    .def("GetCalibrationMatrix", &theia::DivisionUndistortionCameraModel::GetCalibrationMatrix)
    .def("PrintIntrinsics", &theia::DivisionUndistortionCameraModel::PrintIntrinsics)
    .def_property_readonly("kIntrinsicsSize", &theia::DivisionUndistortionCameraModel::NumParameters)
    .def_property("AspectRatio", &theia::DivisionUndistortionCameraModel::AspectRatio, &theia::DivisionUndistortionCameraModel::SetAspectRatio)
    .def_property_readonly("RadialDistortion1", &theia::DivisionUndistortionCameraModel::RadialDistortion1)
    .def("SetRadialDistortion", &theia::DivisionUndistortionCameraModel::SetRadialDistortion)

  ;

  // PinholeCameraModel
  py::class_<theia::PinholeCameraModel, std::shared_ptr<theia::PinholeCameraModel>>(m, "PinholeCameraModel", camera_intrinsics_model)
    .def(py::init<>())
    .def("Type", &theia::PinholeCameraModel::Type)
    .def("NumParameters", &theia::PinholeCameraModel::NumParameters)
    .def("SetFromCameraIntrinsicsPriors", &theia::PinholeCameraModel::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::PinholeCameraModel::CameraIntrinsicsPriorFromIntrinsics)
    // OptimizeIntrinsicsType not defined
    .def("GetSubsetFromOptimizeIntrinsicsType", &theia::PinholeCameraModel::GetSubsetFromOptimizeIntrinsicsType)
    .def("GetCalibrationMatrix", &theia::PinholeCameraModel::GetCalibrationMatrix)
    .def("PrintIntrinsics", &theia::PinholeCameraModel::PrintIntrinsics)
    .def_property_readonly("kIntrinsicsSize", &theia::PinholeCameraModel::NumParameters)
    .def_property("AspectRatio", &theia::PinholeCameraModel::AspectRatio, &theia::PinholeCameraModel::SetAspectRatio)
    .def_property("Skew", &theia::PinholeCameraModel::Skew, &theia::PinholeCameraModel::SetSkew)
    .def_property_readonly("RadialDistortion1", &theia::PinholeCameraModel::RadialDistortion1)
    .def_property_readonly("RadialDistortion2", &theia::PinholeCameraModel::RadialDistortion2)
    .def("SetRadialDistortion", &theia::PinholeCameraModel::SetRadialDistortion)

  ;

  // FOVCameraModel
  py::class_<theia::FOVCameraModel, std::shared_ptr<theia::FOVCameraModel>>(m, "FOVCameraModel", camera_intrinsics_model)
    .def(py::init<>())
    .def("Type", &theia::FOVCameraModel::Type)
    .def("NumParameters", &theia::FOVCameraModel::NumParameters)
    .def("SetFromCameraIntrinsicsPriors", &theia::FOVCameraModel::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::FOVCameraModel::CameraIntrinsicsPriorFromIntrinsics)
    // OptimizeIntrinsicsType not defined
    .def("GetSubsetFromOptimizeIntrinsicsType", &theia::FOVCameraModel::GetSubsetFromOptimizeIntrinsicsType)
    .def("GetCalibrationMatrix", &theia::FOVCameraModel::GetCalibrationMatrix)
    .def("PrintIntrinsics", &theia::FOVCameraModel::PrintIntrinsics)
    .def_property_readonly("kIntrinsicsSize", &theia::FOVCameraModel::NumParameters)
    .def_property("AspectRatio", &theia::FOVCameraModel::AspectRatio, &theia::FOVCameraModel::SetAspectRatio)
    .def_property_readonly("RadialDistortion1", &theia::FOVCameraModel::RadialDistortion1)
    .def("SetRadialDistortion", &theia::FOVCameraModel::SetRadialDistortion)

  ;

  m.def("ComposeProjectionMatrix", theia::ComposeProjectionMatrixWrapper);
  m.def("DecomposeProjectionMatrix", theia::DecomposeProjectionMatrixWrapper);
  m.def("CalibrationMatrixToIntrinsics", theia::CalibrationMatrixToIntrinsicsWrapper);
  m.def("IntrinsicsToCalibrationMatrix", theia::IntrinsicsToCalibrationMatrixWrapper);



  py::class_<theia::Camera, std::shared_ptr<theia::Camera>>(m, "Camera")
    .def(py::init())
    .def(py::init<theia::Camera>())
    .def(py::init<theia::CameraIntrinsicsModelType>())
    .def("CameraIntrinsics", &theia::Camera::CameraIntrinsics, py::return_value_policy::reference)
    .def("SetFromCameraIntrinsicsPriors", &theia::Camera::SetFromCameraIntrinsicsPriors)
    .def("CameraIntrinsicsPriorFromIntrinsics", &theia::Camera::CameraIntrinsicsPriorFromIntrinsics)
    .def("GetCameraIntrinsicsModelType", &theia::Camera::GetCameraIntrinsicsModelType)
    .def("SetCameraIntrinsicsModelType", &theia::Camera::SetCameraIntrinsicsModelType)
    .def("InitializeFromProjectionMatrix", &theia::Camera::InitializeFromProjectionMatrix)
    .def("GetCalibrationMatrix", &theia::Camera::GetCalibrationMatrixWrapper)
    .def("GetProjectionMatrix", &theia::Camera::GetProjectionMatrixWrapper)
    .def_property("FocalLength", &theia::Camera::FocalLength, &theia::Camera::SetFocalLength)
    .def_property_readonly("ImageHeight", &theia::Camera::ImageHeight)
    .def_property_readonly("ImageWidth", &theia::Camera::ImageWidth)
    .def("SetImageSize", &theia::Camera::SetImageSize)
    .def_property_readonly("PrincipalPointX", &theia::Camera::PrincipalPointX)
    .def_property_readonly("PrincipalPointY", &theia::Camera::PrincipalPointY)
    .def("SetPrincipalPoint", &theia::Camera::SetPrincipalPoint)
    .def("GetOrientationAsAngleAxis", &theia::Camera::GetOrientationAsAngleAxis)
    .def("GetOrientationAsRotationMatrix", &theia::Camera::GetOrientationAsRotationMatrix)
    .def("SetOrientationFromAngleAxis", &theia::Camera::SetOrientationFromAngleAxis)
    .def("SetOrientationFromRotationMatrix", &theia::Camera::SetOrientationFromRotationMatrix)
    .def_property("Position", &theia::Camera::GetPosition, &theia::Camera::SetPosition)
    .def("PrintCameraIntrinsics", &theia::Camera::PrintCameraIntrinsics)
    .def("PixelToNormalizedCoordinates", &theia::Camera::PixelToNormalizedCoordinates)
    .def("PixelToUnitDepthRay", &theia::Camera::PixelToUnitDepthRay)
    .def("ProjectPoint", &theia::Camera::ProjectPointWrapper)
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
  m.def("FivePointFocalLengthRadialDistortion", theia::FivePointFocalLengthRadialDistortionWrapper);
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
  m.def("AlignReconstructions", theia::AlignReconstructions);
  m.def("AlignReconstructionsRobust", theia::AlignReconstructionsRobust);



  //triangulation
  m.def("Triangulate", theia::TriangulateWrapper);
  m.def("TriangulateMidpoint", theia::TriangulateMidpointWrapper);
  m.def("TriangulateDLT", theia::TriangulateDLTWrapper);
  m.def("TriangulateNViewSVD", theia::TriangulateNViewSVDWrapper);
  m.def("TriangulateNView", theia::TriangulateNViewWrapper);
  m.def("IsTriangulatedPointInFrontOfCameras", theia::IsTriangulatedPointInFrontOfCameras);
  m.def("SufficientTriangulationAngle", theia::SufficientTriangulationAngle);


  // View class
  py::class_<theia::View>(m, "View")
    .def(py::init<>())
    .def(py::init<std::string>())
    .def_property_readonly("Name", &theia::View::Name)
    .def_property("IsEstimated", &theia::View::IsEstimated, &theia::View::SetEstimated)
    .def("NumFeatures", &theia::View::NumFeatures)
    .def("AddFeature", &theia::View::AddFeature)
    .def("RemoveFeature", &theia::View::RemoveFeature)
    .def_property_readonly("TrackIds", &theia::View::TrackIds)
    //.def_readwrite("focal_length", &theia::View::Track)

    .def("GetFeature", &theia::View::GetFeature, py::return_value_policy::reference)
    .def("Camera", &theia::View::Camera, "Camera class object")
    .def("CameraIntrinsicsPrior", &theia::View::CameraIntrinsicsPrior)

  ;

  // Track class
  py::class_<theia::Track>(m, "Track")
    .def(py::init<>())

    //.def_property_readonly("Name", &theia::View::Name)
    .def_property("IsEstimated", &theia::Track::IsEstimated, &theia::Track::SetEstimated)
    .def("NumViews", &theia::Track::NumViews)
    .def("AddView", &theia::Track::AddView)
    .def("RemoveView", &theia::Track::RemoveView)
    .def_property_readonly("ViewIds", &theia::Track::ViewIds)
    //.def_readwrite("focal_length", &theia::View::Track)

    //.def("GetFeature", &theia::View::GetFeature, py::return_value_policy::reference)
    .def("Point", &theia::Track::Point)
    .def("Color", &theia::Track::Color)

  ;


  // Reconstruction class
  py::class_<theia::Reconstruction>(m, "Reconstruction")
    .def(py::init<>())

    //.def_property("IsEstimated", &theia::Reconstruction::IsEstimated, &theia::Track::SetEstimated)
    .def("NumViews", &theia::Reconstruction::NumViews)
    .def("ViewIdFromName", &theia::Reconstruction::ViewIdFromName)
    //.def("set", static_cast<void (Pet::*)(int)>(&Pet::set), "Set the pet's age")
    //.def("AddView", static_cast<theia::ViewId (theia::Reconstruction*)(const std::string&)>(&theia::Reconstruction::AddView))
    .def("AddView", (theia::ViewId (theia::Reconstruction::*)(const std::string&)) &theia::Reconstruction::AddView, py::return_value_policy::reference_internal)
    .def("AddView", (theia::ViewId (theia::Reconstruction::*)(const std::string&, const theia::CameraIntrinsicsGroupId)) &theia::Reconstruction::AddView, py::return_value_policy::reference_internal)
    .def("RemoveView", &theia::Reconstruction::RemoveView)
    .def_property_readonly("ViewIds", &theia::Reconstruction::ViewIds)

    .def_property_readonly("NumTracks", &theia::Reconstruction::NumTracks)
    .def("AddTrack", (theia::TrackId (theia::Reconstruction::*)()) &theia::Reconstruction::AddTrack, py::return_value_policy::reference_internal)
    .def("AddTrack", (theia::TrackId (theia::Reconstruction::*)(const std::vector<std::pair<theia::ViewId, theia::Feature>>&)) &theia::Reconstruction::AddTrack, py::return_value_policy::reference_internal)

          //.def("AddTrack", &theia::Reconstruction::AddTrack)
    .def("RemoveTrack", &theia::Reconstruction::RemoveTrack)
    .def_property_readonly("TrackIds", &theia::Reconstruction::TrackIds)

    .def("AddObservation", &theia::Reconstruction::AddObservation)
    .def_property_readonly("NumCameraIntrinsicGroups", &theia::Reconstruction::NumCameraIntrinsicGroups)
    .def("Normalize", &theia::Reconstruction::Normalize)
    .def("CameraIntrinsicsGroupIdFromViewId", &theia::Reconstruction::CameraIntrinsicsGroupIdFromViewId)
    .def("CameraIntrinsicsGroupIds", &theia::Reconstruction::CameraIntrinsicsGroupIds)


    .def("View", &theia::Reconstruction::View, py::return_value_policy::reference)
    .def("Track", &theia::Reconstruction::Track, py::return_value_policy::reference)

  ;

  // TwoViewInfo
  py::class_<theia::TwoViewInfo>(m, "TwoViewInfo")
    .def(py::init<>())
    .def_readwrite("focal_length_1", &theia::TwoViewInfo::focal_length_1)
    .def_readwrite("focal_length_2", &theia::TwoViewInfo::focal_length_2)
    .def_readwrite("position_2", &theia::TwoViewInfo::position_2)
    .def_readwrite("rotation_2", &theia::TwoViewInfo::rotation_2)
    .def_readwrite("num_verified_matches", &theia::TwoViewInfo::num_verified_matches)
    .def_readwrite("num_homography_inliers", &theia::TwoViewInfo::num_homography_inliers)
    .def_readwrite("visibility_score", &theia::TwoViewInfo::visibility_score)
  ;

  m.def("SwapCameras", &theia::SwapCameras);


  // ViewGraph
  py::class_<theia::ViewGraph>(m, "ViewGraph")
    .def(py::init<>())
    //.def_property_readonly("Name", &theia::View::Name)
    .def("ReadFromDisk", &theia::ViewGraph::ReadFromDisk)
    .def("WriteToDisk", &theia::ViewGraph::WriteToDisk)
    .def("HasView", &theia::ViewGraph::HasView)
    .def("RemoveView", &theia::ViewGraph::RemoveView)
    .def("HasEdge", &theia::ViewGraph::HasEdge)
    .def("AddEdge", &theia::ViewGraph::AddEdge)
    .def("RemoveEdge", &theia::ViewGraph::RemoveEdge)
    .def_property_readonly("NumViews", &theia::ViewGraph::NumViews)
    .def_property_readonly("NumEdges", &theia::ViewGraph::NumEdges)
    .def("GetNeighborIdsForView", &theia::ViewGraph::GetNeighborIdsForView, py::return_value_policy::reference)
    .def("GetEdge", &theia::ViewGraph::GetEdge, py::return_value_policy::reference)
    .def("GetAllEdges", &theia::ViewGraph::GetAllEdges)

    // not sure pointer as input
    //.def("ExtractSubgraph", &theia::ViewGraph::ExtractSubgraph)
    //.def("GetLargestConnectedComponentIds", &theia::ViewGraph::GetLargestConnectedComponentIds)

  ;




  // GPS converter
  py::class_<theia::GPSConverter>(m, "GPSConverter")
    .def(py::init<>())
    .def_static("ECEFToLLA", theia::GPSConverter::ECEFToLLA)
    .def_static("LLAToECEF", theia::GPSConverter::LLAToECEF)
  ;

  // Bundle Adjustment
  py::enum_<theia::OptimizeIntrinsicsType>(m, "OptimizeIntrinsicsType")
    .value("NONE", theia::OptimizeIntrinsicsType::NONE)
    .value("FOCAL_LENGTH", theia::OptimizeIntrinsicsType::FOCAL_LENGTH)
    .value("ASPECT_RATIO", theia::OptimizeIntrinsicsType::ASPECT_RATIO)
    .value("SKEW", theia::OptimizeIntrinsicsType::SKEW)
    .value("PRINCIPAL_POINTS", theia::OptimizeIntrinsicsType::PRINCIPAL_POINTS)
    .value("RADIAL_DISTORTION", theia::OptimizeIntrinsicsType::RADIAL_DISTORTION)
    .value("TANGENTIAL_DISTORTION", theia::OptimizeIntrinsicsType::TANGENTIAL_DISTORTION)
    .value("ALL", theia::OptimizeIntrinsicsType::ALL)
    .export_values()
  ;

  py::enum_<theia::LossFunctionType>(m, "LossFunctionType")
    .value("TRIVIAL", theia::LossFunctionType::TRIVIAL)
    .value("HUBER", theia::LossFunctionType::HUBER)
    .value("SOFTLONE", theia::LossFunctionType::SOFTLONE)
    .value("CAUCHY", theia::LossFunctionType::CAUCHY)
    .value("ARCTAN", theia::LossFunctionType::ARCTAN)
    .value("TUKEY", theia::LossFunctionType::TUKEY)
    .export_values()
  ;

  py::class_<theia::BundleAdjustmentOptions>(m, "BundleAdjustmentOptions")
    .def(py::init<>())
    .def_readwrite("loss_function_type", &theia::BundleAdjustmentOptions::loss_function_type)
    .def_readwrite("robust_loss_width", &theia::BundleAdjustmentOptions::robust_loss_width)
    .def_readwrite("verbose", &theia::BundleAdjustmentOptions::verbose)
    .def_readwrite("constant_camera_orientation", &theia::BundleAdjustmentOptions::constant_camera_orientation)
    .def_readwrite("constant_camera_position", &theia::BundleAdjustmentOptions::constant_camera_position)
    .def_readwrite("intrinsics_to_optimize", &theia::BundleAdjustmentOptions::intrinsics_to_optimize)
    .def_readwrite("num_threads", &theia::BundleAdjustmentOptions::num_threads)
    .def_readwrite("max_num_iterations", &theia::BundleAdjustmentOptions::max_num_iterations)
    .def_readwrite("max_solver_time_in_seconds", &theia::BundleAdjustmentOptions::max_solver_time_in_seconds)
    .def_readwrite("use_inner_iterations", &theia::BundleAdjustmentOptions::use_inner_iterations)
    .def_readwrite("function_tolerance", &theia::BundleAdjustmentOptions::function_tolerance)
    .def_readwrite("gradient_tolerance", &theia::BundleAdjustmentOptions::gradient_tolerance)
    .def_readwrite("parameter_tolerance", &theia::BundleAdjustmentOptions::parameter_tolerance)
    .def_readwrite("max_trust_region_radius", &theia::BundleAdjustmentOptions::max_trust_region_radius)
  ;

  //TwoViewBundleAdjustmentOptions
  py::class_<theia::TwoViewBundleAdjustmentOptions>(m, "TwoViewBundleAdjustmentOptions")
    .def(py::init<>())
    .def_readwrite("ba_options", &theia::TwoViewBundleAdjustmentOptions::ba_options)
    .def_readwrite("constant_camera1_intrinsics", &theia::TwoViewBundleAdjustmentOptions::constant_camera1_intrinsics)
    .def_readwrite("constant_camera2_intrinsics", &theia::TwoViewBundleAdjustmentOptions::constant_camera2_intrinsics)
  ;

  py::class_<theia::BundleAdjustmentSummary>(m, "BundleAdjustmentSummary")
    .def(py::init<>())
    .def_readwrite("success", &theia::BundleAdjustmentSummary::success)
    .def_readwrite("initial_cost", &theia::BundleAdjustmentSummary::initial_cost)
    .def_readwrite("final_cost", &theia::BundleAdjustmentSummary::final_cost)
    .def_readwrite("setup_time_in_seconds", &theia::BundleAdjustmentSummary::setup_time_in_seconds)
    .def_readwrite("solve_time_in_seconds", &theia::BundleAdjustmentSummary::solve_time_in_seconds)
  ;

  m.def("BundleAdjustPartialReconstruction", theia::BundleAdjustPartialReconstructionWrapper);
  m.def("BundleAdjustReconstruction", theia::BundleAdjustReconstructionWrapper);
  m.def("BundleAdjustView", theia::BundleAdjustViewWrapper);
  m.def("BundleAdjustTrack", theia::BundleAdjustTrackWrapper);

  m.def("BundleAdjustTwoViews", theia::BundleAdjustTwoViewsWrapper);
  m.def("BundleAdjustTwoViewsAngular", theia::BundleAdjustTwoViewsAngularWrapper);
  m.def("OptimizeRelativePositionWithKnownRotation", theia::OptimizeRelativePositionWithKnownRotationWrapper);

  // Bundle Adjuster
  py::class_<theia::BundleAdjuster>(m, "BundleAdjuster")
    // constructor uses pointer of an object as input
    //.def(py::init<theia::BundleAdjustmentOptions, theia::Reconstruction>())

    .def("AddView", &theia::BundleAdjuster::AddView)
    .def("AddTrack", &theia::BundleAdjuster::AddTrack)
    .def("Optimize", &theia::BundleAdjuster::Optimize)
    //.def("SetCameraExtrinsicsParameterization", &theia::BundleAdjuster::SetCameraExtrinsicsParameterization)
    //.def("SetCameraIntrinsicsParameterization", &theia::BundleAdjuster::SetCameraIntrinsicsParameterization)
    //.def("SetCameraExtrinsicsConstant", &theia::BundleAdjuster::SetCameraExtrinsicsConstant)
    //.def("SetCameraPositionConstant", &theia::BundleAdjuster::SetCameraPositionConstant)
    //.def("SetCameraOrientationConstant", &theia::BundleAdjuster::SetCameraOrientationConstant)
    //.def("SetTrackConstant", &theia::BundleAdjuster::SetTrackConstant)
    //.def("SetTrackVariable", &theia::BundleAdjuster::SetTrackVariable)
    //.def("SetCameraSchurGroups", &theia::BundleAdjuster::SetCameraSchurGroups)
    //.def("SetTrackSchurGroup", &theia::BundleAdjuster::SetTrackSchurGroup)
    //.def("AddReprojectionErrorResidual", &theia::BundleAdjuster::AddReprojectionErrorResidual)
  ;


  // Global SfM
  m.def("ComputeTripletBaselineRatios", theia::ComputeTripletBaselineRatiosWrapper);

  // Position Estimator
  py::class_<theia::PositionEstimator>(m, "PositionEstimator")
  ;

  py::class_<theia::LinearPositionEstimator, theia::PositionEstimator>(m, "LinearPositionEstimator")
    .def(py::init<theia::LinearPositionEstimator::Options, theia::Reconstruction>())
    .def("EstimatePositions", &theia::LinearPositionEstimator::EstimatePositions)

  ;

  py::class_<theia::NonlinearPositionEstimator, theia::PositionEstimator>(m, "NonlinearPositionEstimator")
    .def(py::init<theia::NonlinearPositionEstimator::Options, theia::Reconstruction>())
    .def("EstimatePositions", &theia::NonlinearPositionEstimator::EstimatePositions)

  ;

  py::class_<theia::LeastUnsquaredDeviationPositionEstimator, theia::PositionEstimator>(m, "LeastUnsquaredDeviationPositionEstimator")
    .def(py::init<theia::LeastUnsquaredDeviationPositionEstimator::Options>())
    .def("EstimatePositions", &theia::LeastUnsquaredDeviationPositionEstimator::EstimatePositions)

  ;


  py::class_<theia::RotationEstimator>(m, "RotationEstimator")
  ;

  py::class_<theia::RobustRotationEstimator, theia::RotationEstimator>(m, "RobustRotationEstimator")
    .def(py::init<theia::RobustRotationEstimator::Options>())
    //.def("EstimateRotations", &theia::RobustRotationEstimator::EstimateRotations)
    .def("AddRelativeRotationConstraint", &theia::RobustRotationEstimator::AddRelativeRotationConstraint)
    //.def("EstimateRotations", &theia::RobustRotationEstimator::EstimateRotations)

  ;

  py::class_<theia::NonlinearRotationEstimator, theia::RotationEstimator>(m, "NonlinearRotationEstimator")
    .def(py::init<>())
    .def(py::init<double>())
    .def("EstimateRotations", &theia::NonlinearRotationEstimator::EstimateRotations)

  ;

  py::class_<theia::LinearRotationEstimator, theia::RotationEstimator>(m, "LinearRotationEstimator")
    .def(py::init<>())
    .def("AddRelativeRotationConstraint", &theia::LinearRotationEstimator::AddRelativeRotationConstraint)
    //.def("EstimateRotations", &theia::RobustRotationEstimator::EstimateRotations)
    //.def("EstimateRotations", &theia::RobustRotationEstimator::EstimateRotations)

  ;










}
