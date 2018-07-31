# Works in "virtual_environment" 
#  
# - v5: Detection+Tracking(TDL) (2018.07.27)

from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Define some useful tools from filterpy
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
#gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

#import ji_utils_tracking
from ji_utils_tracking_v2 import *

'''
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
#parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
parser.add_argument('--weights', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()
'''

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# UPDATE HERE
use_cuda = True
use_spatial = True

def cv2_demo(net, transform, use_cuda):
    if use_spatial:
        tresh = 0.3 # UPDATE (smaller is more restrict)
    else:
        tresh = 0.25
        
    detected_objs = {} # store by label name
    step = 0
    
    
    def predict(step,frame,detected_objs, obj_tracked):
        # - Also return the bbox of the tracked obj (2018.07.30)
        pt_tracked = [0,0,0,0]
        
        k_reduce = 0.1 # bbox reduction for matching purpose (detected bbox is large...)
        pts = []
        # Start timer
        timer = cv2.getTickCount()
        
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))
        
        if use_cuda:
            xx = xx.cuda()
            y = net(xx)
        else:
            y = net(xx)  # forward pass
        
        detections = y.data # Objects detected in the frame
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()# Detection box
                # Store tracked obj bbox
                if labelmap[i-1]==obj_tracked:
                    pt_tracked = pt
                
                # Check whether object was already detected
                if labelmap[i - 1] not in detected_objs:
                    # Add
                    vxy = [0,0] # assume static obj at beginning
                    detected_objs[labelmap[i-1]] = [myobj(time.time(),frame,1,labelmap[i-1],get_bbox(pt,k_reduce), vxy)]
                    # Draw box
                    cv2.rectangle(frame,
                                  (int(pt[0]), int(pt[1])), # UPPER LEFT
                                  (int(pt[2]), int(pt[3])), # LOWER RIGHT
                                  COLORS[i % 3], 2)
                    cv2.putText(frame, labelmap[i - 1]+'(1)', (int(pt[0]), int(pt[1])),
                                FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    
                else:
                    if labelmap[i-1] == 'person':
                        treshx = 0.5 # Less restrict for people
                    else:
                        treshx = tresh
                    
                    
                   # Reduce bbox for figure matching
                    newobj = myobj(time.time(),frame,0,labelmap[i-1],get_bbox(pt,k_reduce),[0,0])
                    #showme(newobj.get_img())
                    
                    # Check it was detected before
                    #print('Comparing images...')
                    num, match_obj, dist = get_matching(newobj,detected_objs[labelmap[i - 1]], treshx) # Return 0 if not in the list
                    
                    
                    if num != 0: # object found --> estimate the spatial evidence
                        if use_spatial:
                            dt = newobj.get_time() - match_obj.get_time() # Time diff
                            prior_x, prior_y = predict_pos(match_obj,dt)
                            likelihood_x, likelihood_y  = newobj.get_p_xy()
                            x_distrib = update(prior_x, likelihood_x)
                            y_distrib = update(prior_y, likelihood_y)
                            
                            x,y = newobj.get_xy()
                            approx_spatial_diff = abs(x-x_distrib.mean)+abs(y-y_distrib.mean)  
                            # Experimental
                            #spatial_evidence = 1 - approx_spatial_diff/300
                            
                            if (dist+approx_spatial_diff/300) < treshx: # If spatial difference is small, it will keep
                                # udate velocity for the next iteration
                                vxy = []
                                vxy.append((newobj.get_xy()[0]-match_obj.get_xy()[0])/dt)
                                vxy.append((newobj.get_xy()[1]-match_obj.get_xy()[1])/dt)
                                newobj.set_vxy(vxy) # set velocity
                                detected_objs[labelmap[i - 1]][match_obj.get_id()-1] = newobj.set_id(num) # Update with the matched newer object
                                print('-- dist(spatial):%2.2f(dt=%2.2fs)'%(approx_spatial_diff,dt))
                            else:
                                print('\n -- Too far!! New object distance:',approx_spatial_diff,' --')
                                num = len(detected_objs[labelmap[i-1]])+1 # Add as new image as well
                                detected_objs[labelmap[i-1]].append(newobj.set_id(num))
                                
                        
                    else: # Add if the same image was never seem before
                        num = len(detected_objs[labelmap[i-1]])+1
                        detected_objs[labelmap[i-1]].append(newobj.set_id(num))
                        print(' -- New object added: %s...'%(labelmap[i-1]))
                    
                    cv2.rectangle(frame,
                                (int(pt[0]), int(pt[1])), # UPPER LEFT
                                (int(pt[2]), int(pt[3])), # LOWER RIGHT
                                COLORS[i % 3], 2)
                    
                    # Print info
                    # Calculate Frames per second (FPS)
                    fps_current = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                    score = detections[0,i,j,0]
                    #cv2.putText(frame, labelmap[i - 1]+'('+str(num)+'): score:'+str(int(100* score)) +' (fps:'+ str(int(fps_current)) +')', (int(pt[0]), int(pt[1])),
                    #            FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, labelmap[i - 1]+': score:'+str(int(100* score)) +' (fps:'+ str(int(fps_current)) +')', (int(pt[0]), int(pt[1])),
                                FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    
                j += 1
                pts.append(pt)
        
        if not pts:
            print(' -- No detection (score < 60)...')
            
        return frame, pts, detected_objs, pt_tracked
    
    def detect_tracked(frame,bbox_tracked):
        obj_tracked = ''
        
        bb = bbox_tracked
        
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))
        
        if use_cuda:
            xx = xx.cuda()
            y = net(xx)
        else:
            y = net(xx)  # forward pass
        
        detections = y.data # Objects detected in the frame
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()# Detection box
            while detections[0, i, j, 0] >= 0.6:
                # Check whether bbox_tracked is within the detected obj
                if bb[0]>pt[0] and bb[1]>pt[1] and (bb[0]+bb[2])<pt[2] and (bb[1] + bb[3])<pt[3]:
                    obj_tracked = labelmap[i-1]
                    
                j += 1
                
        if not obj_tracked:
            print(' -- Tracked object not detected (score < 60)...')
            obj_tracked = ''
            
        return obj_tracked
    
    
    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
    
    # grab frame for TDL selection
    frame = stream.read()
    # Select object to track
    bbox = cv2.selectROI(frame, False)
    
    # Define the tracker
    tracker = cv2.TrackerTLD_create()
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    # Identify tracked object
    obj_tracked = detect_tracked(frame,bbox)
    print('Tracked object:', obj_tracked)
    
    # start fps timer
    # loop over frames from the video file stream
    while True:
        step += 1
        # grab next frame
        frame = stream.read()
        key = cv2.waitKey(1) & 0xFF
        # update FPS counter
        fps.update()
        #frame0 = frame
        
        frame, pts, objs, pt_tracked = predict(step,frame, detected_objs,obj_tracked)
        print('Step:',step,end='')
        
        
        
        # Update tracker
        ok, bbox = tracker.update(frame)
        delta = 10 # delta to amplify bbox of detected obj
         
        # Draw bounding box
        if ok and bbox[0]>pt_tracked[0]-delta and bbox[1]>pt_tracked[1]-delta and (bbox[0]+bbox[2])<pt_tracked[2]+delta and (bbox[1] + bbox[3])<pt_tracked[3]+delta:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
  
        
        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # Esc
            break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 21)    # initialize SSD
#    net.load_state_dict(torch.load(args.weights))
    net.load_state_dict(torch.load('weights/ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    fps = FPS().start()
    
    if use_cuda:
        net = net.cuda() # Jaime: convey the use of GPU
        
    cv2_demo(net.eval(), transform, use_cuda)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()      
    #stream.stop()
