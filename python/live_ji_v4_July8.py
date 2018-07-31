# Works in "virtual_environment" 

# - v4: Detection + Track Multiple (2018.07.27)
#
# Key features of SSD:
# - 1) Single-shot: you look only once
# - 2) Multiboxes: classify within the boxes --> speed
# - 3) Multiscale classification layers --> accuracy

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
    
    def predict(step,frame,detected_objs):
        k_reduce = 0.1 # bbox reduction for matching purpose (detected bbox is large...)
        pts = []
        
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
                    
                    score = detections[0,i,j,0]
                    cv2.putText(frame, labelmap[i - 1]+'('+str(num)+'):'+str(int(100* score)), (int(pt[0]), int(pt[1])),
                                FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    
                j += 1
                pts.append(pt)
        
        if not pts:
            print(' -- No detection (score < 60)...')
            
        return frame, pts, detected_objs

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
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
        frame, pts, objs = predict(step,frame, detected_objs)
        print('Step:',step,end='')

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
