import numpy as np
import cv2 
#from camera_calibration import mtx, dist
import glob

#  pytheia camera module
from pytheia_sfm import Camera, PinholeRadialTangentialCameraModel, CameraIntrinsicsPrior, CameraIntrinsicsModeType

# 
from pytheia_image import Keypoint, KeypointType
from pytheia_matching import KeypointsAndDescriptors


# pytheia feature matcher
from pytheia_matching import InMemoryFeaturesAndMatchesDatabase, BruteForceFeatureMatcher

# FeatureMatcher options
from pytheia_matching import FeatureMatcherOptions
from pytheia_sfm import TwoViewMatchGeometricVerification

from pytheia_sfm import EstimateTwoViewInfoOptions, EstimateTwoViewInfo, TwoViewInfo

from pytheia_sfm import RansacType

from pytheia_matching import FeatureCorrespondence, IndexedFeatureMatch, ImagePairMatch

from pytheia_sfm import ViewGraph, Reconstruction, TrackBuilder, LossFunctionType, GlobalReconstructionEstimator, ReconstructionEstimatorSummary

# Reconstruction related import 
from pytheia_matching import MatchingStrategy
from pytheia_image import FeatureDensity
from pytheia_image import DescriptorExtractorType
from pytheia_matching import FeatureMatcherOptions
from pytheia_sfm import ReconstructionEstimatorOptions
from pytheia_sfm import ReconstructionBuilderOptions, ReconstructionBuilder

from pytheia_io import WritePlyFile, WriteReconstruction

def match_image_pair(image_file_path1, image_file_path2):

    img1_name = remove_prefix_and_suffix(image_file_path1)
    img2_name = remove_prefix_and_suffix(image_file_path2)
    view_id1 = recon.ViewIdFromName(img1_name)
    view_id2 = recon.ViewIdFromName(img2_name)
    img1 = cv2.imread(image_file_path1)
    img2 = cv2.imread(image_file_path2)

    akaze = cv2.AKAZE_create()
    #orb = cv2.ORB()
    #kp1, des1 = orb.detectAndCompute(img1,None)
    #kp2, des2 = orb.detectAndCompute(img2,None)
    #sift = cv2.SIFT()
    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)   

    pts1 = [p.pt for p in kpts1]
    pts2 = [p.pt for p in kpts2]

    # create BFMatcher object

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    #matches = bf.knnMatch(desc1,desc2, k=2)

    # Apply ratio test
    filtered_matches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            filtered_matches.append(m)

    print('Number of putative matches: {}'.format(len(filtered_matches)))

    if len(filtered_matches) < min_num_inlier_matches:
        print('Number of putative matches too low!')
        return False, None

    correspondences = correspondence_from_indexed_matches(filtered_matches, pts1, pts2)

    options = EstimateTwoViewInfoOptions()
    success, twoview_info, inlier_indices = EstimateTwoViewInfo(options, prior, prior, correspondences)

    print('Only {} matches survived after geometric verification'.format(len(inlier_indices)))
    if len(inlier_indices) < min_num_inlier_matches:
        print('Number of putative matches after geometric verification are too low!')
        return False, None

    else:
        verified_matches = []
        for i in range(len(inlier_indices)):
            verified_matches.append(filtered_matches[inlier_indices[i]])

        correspondences = correspondence_from_indexed_matches(verified_matches, pts1, pts2)
        twoview_info.num_verified_matches = len(verified_matches)
        imagepair_match = ImagePairMatch()
        imagepair_match.image1 = img1_name
        imagepair_match.image2 = img2_name
        imagepair_match.twoview_info = twoview_info
        imagepair_match.correspondences = correspondences

        for i in range(len(verified_matches)):
            track_builder.AddFeatureCorrespondence(view_id1, correspondences[i].feature1, view_id2, correspondences[i].feature2)
        
        return True, imagepair_match


def remove_prefix_and_suffix(image_file):
    if image_file.endswith('.jpeg'):
        image_name = image_file[7:-5]
    
    return image_name

min_num_inlier_matches = 30

# create correspondences of keypoints locations from indexed feature matches
def correspondence_from_indexed_matches(filtered_matches, pts1, pts2):
    correspondences = []
    for match in filtered_matches:
        point1 = np.array(pts1[match.queryIdx])
        point2 = np.array(pts2[match.trainIdx])
        feature_correspondece = FeatureCorrespondence(point1, point2)
        correspondences.append(feature_correspondece)

    return correspondences

def match_features_opencv(images_files, images_and_keypoints, images_and_descriptors):

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1,desc2)

    #descriptor format numpy array num_features * feature vector length
    

def match_features_theia(images_files):

    fm_options = FeatureMatcherOptions()
    fm_options.num_threads = 1
    fm_options.perform_geometric_verification = True
    fm_options.geometric_verification_options.bundle_adjustment = True
    fm = BruteForceFeatureMatcher(fm_options, features_and_matches_db)
    fm.AddImages(images_files)
    print('Matching images...')
    fm.MatchImages()


def extract_features_opencv(images_files):
    akaze = cv2.AKAZE_create()
    images_and_descriptors = {}
    images_and_keypoints = {}
    for image_name in images_files:

        img = cv2.imread(image_name)
        kpts1, desc1 = akaze.detectAndCompute(img, None)
        pts1 = [p.pt for p in kpts1]

        type = KeypointType(2) # keypoint type is akaze
        #image_features = KeypointsAndDescriptors()
        #image_features.image_name = image_name
        keypoints = []
        descriptors = []
        for i in range(len(pts1)):
            x = pts1[i][0]
            y = pts1[i][1]
            keypoint = Keypoint(x, y, type)
            keypoints.append(keypoint)
            descriptors.append(desc1[i])
        print('{} keypoints detected in image {}'.format(len(keypoints), image_name))
        images_and_keypoints[image_name] = keypoints
        images_and_descriptors[image_name] = descriptors
        #image_features.keypoints = keypoints
        #image_features.descriptors = descriptors
        #features_and_matches_db.PutFeatures(image_name, image_features)
        #features_and_matches_db.PutCameraIntrinsicsPrior(image_name, prior)

    return images_and_keypoints, images_and_descriptors
    

def extract_features_theia(images_files):

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

    return features_and_matches_db




if __name__ == "__main__":
    #Pipeline starts
    print('Pipeline starts...')

    view_graph = ViewGraph()
    recon = Reconstruction()
    track_builder = TrackBuilder(3, 30)

    # camera intrinsic parameters and distortion coefficients
    cam_matrix = [[3.06661152e+03, 0.00000000e+00, 1.96956075e+03],
    [0.00000000e+00, 3.07017469e+03, 1.57371386e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    distortion_params = [0.13437347, -0.83580167,  0.00354001,  0.00444744,  1.33033269]
    k1, k2, p1, p2, k3 = distortion_params
    print('distortion coefficients are: {}{}{}{}{}'.format(k1, k2, p1, p2, k3))


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

    # pinhole radial tangential camera
    camera = Camera(CameraIntrinsicsModeType(1))
    camera.SetFromCameraIntrinsicsPriors(prior)


    # opencv extraction of features from images
    images_files = glob.glob('milk/*.jpeg')
    image_names = []
    for image_file in images_files:
        image_names.append(remove_prefix_and_suffix(image_file))
    
    print('{} images have been found'.format(len(images_files)))

    for image_name in image_names:
        view_id = recon.AddView(image_name, 0)
        c = recon.MutableView(id).MutableCamera()
        c.DeepCopy(camera)
        recon.MutableView(id).MutableCameraIntrinsicsPrior() = c.CameraIntrinsicsPriorFromIntrinsics()

    #features_and_matches_db = extract_features_theia(images_files)
    #images_and_keypoints, images_and_descriptors = extract_features_opencv(images_files)

    #match_features_opencv(images_and_keypoints, images_and_descriptors, images_files)
    #match_features_theia(images_files)

    num_images = len(images_files)
    for i in range(num_images):
        for j in range(i+1, num_images):
            success, imagepair_match = match_image_pair(images_files[i], images_files[j])
            if success == True:
                view_graph.AddEdge(1, 2, imagepair_match.twoview_info)
            else:
                print("No match between image {} and image {}. ".format(i, j))
    
    track_builder.BuildTracks(recon)
    options = ReconstructionEstimatorOptions()
    options.num_threads = 7
    options.rotation_filtering_max_difference_degrees = 10.0
    options.bundle_adjustment_robust_loss_width = 3.0
    options.bundle_adjustment_loss_function_type = LossFunctionType(1)
    options.subsample_tracks_for_bundle_adjustment = True
    options.filter_relative_translations_with_1dsfm = True


    global_estimator = GlobalReconstructionEstimator(options)
    recon_sum = global_estimator.Estimate(view_graph, recon)

    #std::cout << recon_sum.message << "\n";
    WritePlyFile("test.ply", recon, 2)
    WriteReconstruction(recon, "reconstruction_file")
