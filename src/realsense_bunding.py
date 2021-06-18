#!/usr/bin/python
# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco
import cv2

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=2,
        markersY=2,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

# Start streaming
cfg = pipeline.start(config)
dev = cfg.get_device()

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None


iteration = 0
preset = 0
preset_name = ''

cap = cv2.VideoCapture(0)

try:
    while(True):

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        images = color_image

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(images, preset_name,(720,1300), font, 4,(255,255,255),2,cv2.LINE_AA)

        # Show images
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #ret, RealSense = cap.read()

        # grayscale image
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        
        # Make sure all 5 markers were detected before printing them out
        if ids is not None and len(ids) == 1:
            # Print corners and ids to the console
            for i, corner in zip(ids, corners):
                print('ID: {}; Corners: {}'.format(i, corner))

            # Outline all of the markers detected in our image
            images = aruco.drawDetectedMarkers(images, corners, borderColor=(0, 0, 255))

        #cv2.imshow('RealSense', images)
        cv2.imshow('RealSense', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('t'):
            cv2.imwrite('/home/emol/Desktop/realsense_boundingbox.png',images)
        #result = False
        
        
        #cv2.imshow('frame', frame)
        if key & 0xFF == ord('q') or key == 27:
            break
        
    
    img1 = cv2.imread('/home/emol/Desktop/realsense_boundingbox.png')
    # img2 = img1
    # img3 = cv2.imread('/home/emol/Desktop/test290.png',0)
    # Contours_bounding_box(img1)
    # angle_detect(img2,img3,1)
    

finally:
    cap.release()
    cv2.destroyAllWindows()
    # Stop streaming
    pipeline.stop()

