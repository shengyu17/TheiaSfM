import numpy as np
import cv2 
#from camera_calibration import mtx, dist
import glob

#  pytheia camera module
from pytheia_sfm import Camera, PinholeRadialTangentialCameraModel, CameraIntrinsicsPrior

# 
from pytheia_image import Keypoint, KeypointType
from pytheia_matching import KeypointsAndDescriptors

# pytheia feature matcher
from pytheia_matching import InMemoryFeaturesAndMatchesDatabase, BruteForceFeatureMatcher

# FeatureMatcher options
from pytheia_matching import FeatureMatcherOptions
from pytheia_sfm import TwoViewMatchGeometricVerification
from pytheia_sfm import EstimateTwoViewInfoOptions
from pytheia_sfm import RansacType

#Pipeline starts
print('Pipeline starts...')

# camera intrinsic parameters and distortion coefficients
mtx = [[3.06661152e+03, 0.00000000e+00, 1.96956075e+03],
[0.00000000e+00, 3.07017469e+03, 1.57371386e+03],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_params = [0.13437347, -0.83580167,  0.00354001,  0.00444744,  1.33033269]
cam_matrix = mtx


k1, k2, p1, p2, k3 = distortion_params
print('coefficients are: {}{}{}{}{}'.format(k1, k2, p1, p2, k3))


focal_length = cam_matrix[0][0]
aspect_ratio = cam_matrix[1][1]/cam_matrix[0][0]
cx = cam_matrix[0][2]
cy = cam_matrix[1][2]

prior = CameraIntrinsicsPrior()
prior.focal_length.value = [focal_length]
prior.aspect_ratio.value = [aspect_ratio]
prior.principal_point.value = [cx, cy]
prior.radial_distortion.value = [k1, k2, k3, 0]
prior.tangential_distortion.value = [p1, p2]
prior.skew.value = [0]
prior.camera_intrinsics_model_type = 'PINHOLE_RADIAL_TANGENTIAL'


# opencv extraction of features from images
images_files = glob.glob('images/*.jpeg')

images = []
for fname in images_files:
    img = cv2.imread(fname)
    images.append(img)
print(len(images))

features_and_matches_db = InMemoryFeaturesAndMatchesDatabase()


akaze = cv2.AKAZE_create()
#orb = cv2.ORB()
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)
#sift = cv2.SIFT() 

for image_name in images_files:

    img = cv2.imread(image_name)
    kpts1, desc1 = akaze.detectAndCompute(img, None)
    pts1 = [p.pt for p in kpts1]

    type = KeypointType(2) # keypoint type is akaze
    image_features = KeypointsAndDescriptors()
    image_features.image_name = image_name
    keypoints = []
    descriptors = []
    for i in range(len(pts1)):
        x = pts1[i][0]
        y = pts1[i][1]
        keypoint = Keypoint(x, y, type)
        keypoints.append(keypoint)
        descriptors.append(desc1[i])
    print('{} keypoints detected in image {}'.format(len(keypoints), image_name))
    image_features.keypoints = keypoints
    image_features.descriptors = descriptors
    features_and_matches_db.PutFeatures(image_name, image_features)
    features_and_matches_db.PutCameraIntrinsicsPrior(image_name, prior)


fm_options = FeatureMatcherOptions()
fm_options.num_threads = 1
fm_options.perform_geometric_verification = True
fm_options.geometric_verification_options.bundle_adjustment = True
fm = BruteForceFeatureMatcher(fm_options, features_and_matches_db)
fm.AddImages(images_files)
print('Matching images...')
fm.MatchImages()
