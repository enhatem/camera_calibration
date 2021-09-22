import numpy as np
import cv2 as cv
import glob

###################### FIND CHESSBOARD CORNERS - objPoinrs AND imgPoints ######################

chessboardSize = (24,17) # width,height of the corners in the chess board
frameSize = (1440,1080)  # image resoltion

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objPoints = [] # 3d points in real world space
imgPoints = [] # 2d points in image plane

images = glob.glob('*.png')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True: 

        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        # Draw and display the corners 
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

# print("Object Points: ", objPoints)
# print("Image Points: ", imgPoints)

############################## CALIBRATION ##############################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("Camera Calibrated: ", ret)
print("\nCamera Matrix:\n ", cameraMatrix)
print("\nDistortion Parameters:\n ", dist)
print("\Rotation Vectors:\n ", rvecs)
print("\Translation Vectors:\n ", tvecs)


############################## UNDISTORTION ##############################

img = cv.imread('5.png')
print(img)
h,w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort 
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

############################### 2ND METHOD ###############################

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)


# Reprojection Error
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error += error

print("\ntotal error: {}".format(mean_error/len(objPoints)))
print("\n\n\n")