# 3dReconstruction
Expoloring the world of generating 3d models from images

## Homography.ipynb
Homography assumes that all matched points are planar. Try to leverage this to stitch images together. The results are better than expected, but not great

## StereoDepthMap.ipynb
Use OpenCV to determine a depthmap for 2 images.

## CameraPositions.ipynb
Finding the essential or fundamental matrix determining the orientation of 2 images

## PoseEstimation.ipynb
Determine the pose of a camera relative to a checkerboard and draw a model on it. Shows how to use PnPRansac to determine rotation and translation from checkerboard, and how to place a 3d model on the image in the correct orientation and scale.  Also covers how to scale the object to real world dimensions based on the checkerboard size.

## Essential Matrix and Stereo Reconstruction
* Given 2 images, find the essential matrix for them (relates the 2nd image to the first)
* Use the essential matrix and points to determine the pose (physical rotation and translation at unknown scale) from one to the next.
* Then can rectify the images so that their epilines are colinear. 
  * The stereo rectify algorthim in opencv rectifies doing the following (on the camera intrincs and extrinsics):
    * Scales the rotation vector by half and applies it to both images which spreads distortion across both of the images
    * Handles a bunch of edge cases for when the output image is not the same as the input image
    * Calculates the cx and cy values for the cameras after the images have been rectified using the inlier points
    * Calculates the reprojection matrix Q
* Actually do the rectification on the images using rectifyMap
* Compute disparity using StereoSGBM
* Filter out bad points 
  * Not in disparity map
  * Are parts of small clusters
* Reproject the image to 3d (at unknown scale)
* Save the 3d points and their colors to a ply file



# Multi-View Stereo
Building a pipeline for creating 3d models from multiple images. Trying to leverage the pipeline from the stereo reconstruction notebook does not extend easily to using multiple images. Specifically the way that it rectifies the images makes the merging of multiple stereo pairs together very difficult becuase it rectifies the images by moving both images to the the center of the the rotation in which adds complexity. The stereo rectify code is also very generic and handles a lot of cases that I am not particularly interested in, such as vertical stereo, different left and right image sizes, different output image size, different left and right calibration matricies. So will be building a pipeline that makes the following assumptions:
1. The same camera is used for all images, and the camera matrix and distortion coeffs are the same
2. All images are taken in horizontal disparity, ie the camera is shifted left to right, not up down. While moving around the object the camera may shift slightly up and down in each image, the dominate movement is to be horizontal. Not sure if I will handle 'loop closure' where one image is mostly in vertical stereo to another
3. The movement between 2 images is small enough that we can rectify one image and still have sufficient resolution. This one is huge to simplify the creation of the re-projection matrix into one of the cameras coordinate system. If both images need to be rectified then the reprojection matrix has to change for each image pair. If only one image needs to be rectified, then every disparity map created can use the same reprojection matrix (one for horizontal disparities and one for vertical disparities)
   * If this is true then can have reprojection matrix for horizontal disparity be 
        $$Q_{HorizontalDisparity} = \begin{bmatrix}
                1 & 0 & 0 & -cx_{LeftCamera}\\
                0 & 1 & 0 & -cy_{LeftCamera}\\
                0 & 0 & 1 & fx_{LeftCamera}\\
                0 & 0 & 0 & cx_{LeftCamera} - cxRectified_{RightCamera}
                \end{bmatrix} $$
        where `cx` and `fx` come from the camera calibration matrix, and cxRectified is the `cx` value of the right camera after it has been rectified
    
        For Vertical Disparity where bottom camera is the one that is rectified to the top
        $$Q_{VericalDisparity} = \begin{bmatrix}
                1 & 0 & 0 & -cx_{TopCamera}\\
                0 & 1 & 0 & -cy_{TopCamera}\\
                0 & 0 & 1 & fy_{TopCamera}\\
                0 & 0 & 0 & cy_{TopCamera} - cyRectified_{BottomCamera}
                \end{bmatrix} $$
    * 
4. 

