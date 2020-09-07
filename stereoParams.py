import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
image_scale = 0.25

def nothing(x):
    pass
def load_images():

    image_left = cv2.imread("images/saw/saw_01.jpg")
    image_right = cv2.imread("images/saw/saw_02.jpg")
    image_left = cv2.resize(image_left,(0,0),fx=image_scale,fy=image_scale)
    image_right = cv2.resize(image_right,(0,0),fx=image_scale,fy=image_scale)
    return image_left, image_right

def homography(image_left, image_right):

    orb = cv2.ORB_create(nfeatures=1000)

    kps1, descs1 = orb.detectAndCompute(left_gray,None)
    kps2, descs2 = orb.detectAndCompute(right_gray,None)

    MIN_MATCH_COUNT = 10
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number =12, # 12
                    key_size = 10,     # 20
                    multi_probe_level = 2) #2
    search_params = {"checks":100}

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descs1, descs2, k=2)

    # keep good matches
    good = [m for m,n in matches if m.distance<0.7*n.distance]

    if len(good) > MIN_MATCH_COUNT:
        src_points = np.array([ kps2[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1,1,2)
        dst_points = np.array([ kps1[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1,1,2)
        
        M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        image_right_homography = cv2.warpPerspective(image_right,M,(image_left.shape[1],image_left.shape[0]))
    else:
        print("Not enough matches found")
        image_right_homography = None

    return image_right_homography

def stereo_rectification(left_gray,right_gray,parameters):

    orb = cv2.ORB_create(nfeatures=1000)

    kps1, descs1 = orb.detectAndCompute(left_gray,None)
    kps2, descs2 = orb.detectAndCompute(right_gray,None)

    MIN_MATCH_COUNT = 10
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number =12, # 12
                    key_size = 10,     # 20
                    multi_probe_level = 2) #2
    search_params = {"checks":100}

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descs1, descs2, k=2)


    good_matches = []
    pts1 = []
    pts2 = []
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            good_matches.append(m)
            pts2.append(kps2[m.trainIdx].pt)
            pts1.append(kps1[m.queryIdx].pt)

    # Get the all the intrinsic parameters
    mtx = np.array(parameters['mtx'],dtype=np.float32)
    newCameraMtx = np.array(parameters['newCameraMtx'],dtype=np.float32)
    dist = np.array(parameters['dist'],dtype=np.float32)
    roi = np.array(parameters['roi'],dtype=np.int32)
    
    # Now find the essential Matrix
    pts1 = np.array(pts1,dtype=np.int32)
    pts2 = np.array(pts2,dtype=np.int32)
    E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.FM_LMEDS)
    # We select only inlier points
    pts1_inliers = pts1[mask.ravel()==1]
    pts2_inliers = pts2[mask.ravel()==1]

    retval, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, mtx)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx, dist, mtx, dist, (800,1000), R,t)

    # Now correct the images
    shape = (image_left.shape[1],image_left.shape[0])
    mapx1,mapy1  = cv2.initUndistortRectifyMap(mtx, dist, R1, P1,shape, cv2.CV_32FC1)
    mapx2,mapy2  = cv2.initUndistortRectifyMap(mtx, dist, R2, P2,shape, cv2.CV_32FC1)
    image_left_rectified = cv2.remap(image_left,mapx1,mapy1,cv2.INTER_LINEAR)
    image_right_rectified = cv2.remap(image_right,mapx2,mapy2,cv2.INTER_LINEAR)
    
    return image_left_rectified, image_right_rectified

def compute_stereo(image_left,
                    image_right,
                    minDisparity=0,
                    numDisparities=16,
                    blockSize=3,
                    P1=0,
                    P2=0,
                    disp12MaxDiff=0,
                    preFilterCap=0,
                    uniquenessRatio=0,
                    speckleWindowSize=0,
                    speckleRange=0):
    start_time = time.time()
    stereo = cv2.StereoSGBM_create(minDisparity,
                                numDisparities,
                                blockSize,
                                P1,
                                P2,
                                disp12MaxDiff,
                                preFilterCap,
                                uniquenessRatio,
                                speckleWindowSize,
                                speckleRange)
    create_time = time.time() - start_time

    disparity = stereo.compute(image_left,
                            image_right).astype(np.float32)
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    disparity = (255*disparity).astype(np.uint8)

    compute_time = time.time() - start_time - create_time
    elapsed_time = time.time() - start_time
    fps = 1/ elapsed_time
    print(f"Update time: {elapsed_time:.4f} create: {create_time:.6f} compute: {compute_time:.6f} fps: {fps:.1f}")

    return disparity

def update(_):
    minDisparity = cv2.getTrackbarPos('minDisparity','image')
    numDisparities = 16*cv2.getTrackbarPos('numDisparities','image')
    blockSize = cv2.getTrackbarPos('blockSize','image')
    if blockSize%2 == 0:
        blockSize += 1
        cv2.setTrackbarPos("blockSize","image",blockSize)
    P1 = cv2.getTrackbarPos('P1','image')
    P2 = cv2.getTrackbarPos('P2','image')
    if P2 < P1:
        P2 = P1+1
        cv2.setTrackbarPos("P2","image",P2)
    disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff","image")
    preFilterCap = cv2.getTrackbarPos("preFilterCap","image")
    uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio","image")
    speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize","image")
    speckleRange = cv2.getTrackbarPos("speckleRange","image")
    
    disparity = compute_stereo(left_rectified, 
                                    right_rectified,
                                    minDisparity=minDisparity,
                                    numDisparities=numDisparities,
                                    blockSize=blockSize,
                                    P1=P1,
                                    P2=P2,
                                    disp12MaxDiff=disp12MaxDiff,
                                    preFilterCap=preFilterCap,
                                    uniquenessRatio=uniquenessRatio,
                                    speckleWindowSize=speckleWindowSize,
                                    speckleRange=speckleRange)
    
    
    cv2.imshow('image',disparity)


parameter_filename = f"camera_parameters_scale{image_scale:.4f}.json"
with open(parameter_filename,'r') as fp:
    parameters = json.load(fp)

image_left, image_right = load_images()
left_gray = cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
left_rectified, right_rectified = stereo_rectification(left_gray, right_gray,parameters)
#right_homography_gray = homography(left_gray, right_gray)

disparity = compute_stereo(left_rectified, right_rectified)

cv2.namedWindow('image')
cv2.createTrackbar('minDisparity','image',1,255,update)
cv2.createTrackbar('numDisparities','image',1,20,update)
cv2.createTrackbar('blockSize','image',1,21,update)
cv2.createTrackbar('P1','image',0,1500,update)
cv2.createTrackbar('P2','image',0,1500,update)
cv2.createTrackbar('disp12MaxDiff','image',-1,400,update)
cv2.createTrackbar('preFilterCap','image',0,255,update)
cv2.createTrackbar('uniquenessRatio','image',0,255,update)
cv2.createTrackbar('speckleWindowSize','image',0,500,update)
cv2.createTrackbar('speckleRange','image',1,20,update)


update(None)

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()