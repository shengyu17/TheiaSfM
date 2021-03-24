#import the python binding function from where it is defined

from pytheia.pytheia_sfm import CameraIntrinsicsPrior
from pytheia.pytheia_sfm import CameraIntrinsicsModelType, Camera
from pytheia.pytheia_sfm import FisheyeCameraModel



#Setting Camera Intrinsics
cx, cy = 800, 600
k1, k2, k3 = 0, 0, 0
p1, p2 = 0, 0
focal_length = 2000
aspect_ratio = 1.01
prior = CameraIntrinsicsPrior()
prior.focal_length.value = [focal_length]
prior.aspect_ratio.value = [aspect_ratio]
prior.principal_point.value = [cx, cy]
prior.radial_distortion.value = [k1, k2, k3, 0]
prior.tangential_distortion.value = [p1, p2]
prior.skew.value = [0]
prior.camera_intrinsics_model_type = 'PINHOLE_RADIAL_TANGENTIAL'

#Camera Intrinsics Model
intrinsics = FisheyeCameraModel()
print(intrinsics.Type())

#Initializing a Camera object
fisheye = CameraIntrinsicsModelType(2)
pinhole = CameraIntrinsicsModelType(0)
camera1 = Camera(fisheye)
camera2 = Camera(pinhole)
camera1.SetFromCameraIntrinsicsPriors(prior)
prior = camera1.CameraIntrinsicsPriorFromIntrinsics()
camera1.PrintCameraIntrinsics()



