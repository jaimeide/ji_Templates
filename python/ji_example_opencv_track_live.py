# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 22:04:45 2018

@author: jaime

ref: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
"""

import cv2
import sys

import time
from imutils.video import FPS, WebcamVideoStream
 
#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')￼
major_ver, minor_ver, subminor_ver = cv2.__version__.split('.')

if __name__ == '__main__' :
    # Track FPS
    fps_global = FPS().start()
    
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[3]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    
    go_live = True
    if go_live:
        # start video stream thread, allow buffer to fill
        print("[INFO] starting threaded video stream...")
        stream = WebcamVideoStream(src=0).start()  # default camera
        time.sleep(1.0)
        # first frame
        frame = stream.read()
    else:
        # Read video
        video = cv2.VideoCapture("chaplin.mp4")
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
     
        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
    
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    
    # Select object to track
    bbox = cv2.selectROI(frame, False)
     
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    while True:
        if go_live:
            # grab next frame
            frame = stream.read()
        else:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break
        
        key = cv2.waitKey(1) & 0xff
        
        '''
        # Pause and select bbox
        if key == ord('p'):  # pause
            bbox = cv2.selectROI(frame, False)
            ok = tracker.init(frame, bbox) # re-start tracker
        '''
        # Exit if ESC pressed
        if key == 27 : break
    
        # update FPS counter
        fps_global.update()
    
        
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
    
    # stop the timer and display FPS information
    fps_global.stop()
    print("[INFO] elasped time: {:.2f}".format(fps_global.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps_global.fps()))
    
    # cleanup
    cv2.destroyAllWindows()