# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

import matplotlib.pyplot as plt
import numpy as np
# Define some useful tools from filterpy
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
#gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])
from ji_utils_tracking_v2 import *

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Defining a function that will do the detections
def detect(frame, net, transform, step, detected_objs): # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2] # We get the height and the width of the frame.
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    
    xx = Variable(x.unsqueeze(0))
    if use_cuda:
        xx = xx.cuda()
        y = net(xx)
    else:
        y = net(xx)  # forward pass    
    #x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch.
    #y = net(x) # We feed the neural network ssd with the image and we get the output y.
    
    # Tracking part
    k_reduce = 0.1 # bbox reduction for matching purpose (detected bbox is large...)
    pts = []
        
    detections = y.data # We create the detections tensor contained in the output y. 
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].
    # detections (organization): [batch, #class, #occurrence, [score,x0,y0,x1,y1]]
    for i in range(detections.size(1)): # For every class:
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        # 0: is for the batch, here is just a dummy var
        # i: class index
        # j: will store the occurrence
        # 0: get the first element (score) of the 4th array
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            # COORDINATES!
          
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
#            # DRAW A RECTANGLE     **   Lower cornet **       ** Upper corner **     **Red color  **Thickness
#            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
#            # PLACE LABEL TEXT  **Label               **Position               **FONT              **Thickness&COLOR  **LINE
#            score = detections[0, i, j, 0]
#            cv2.putText(frame, labelmap[i - 1]+':'+str(int(100* score)), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
#            j += 1 # We increment j to get to the next occurrence.

            # Check whether object was already detected
            if labelmap[i - 1] not in detected_objs:
                # Add
                vxy = [0,0] # assume static obj at beginning
                detected_objs[labelmap[i-1]] = [myobj(step,frame,1,labelmap[i-1],get_bbox(pt,k_reduce), vxy)]
                # Draw box
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])), # UPPER LEFT
                              (int(pt[2]), int(pt[3])), # LOWER RIGHT
                              COLORS[i % 3], 2)
                score = detections[0, i, j, 0]
                cv2.putText(frame, labelmap[i - 1]+'(1):'+str(int(100* score)), (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
            else:
                # Check it was detected before
                newobj = myobj(step,frame,0,labelmap[i-1],get_bbox(pt,k_reduce),[0,0])
                #print('Comparing images...')
                num, match_obj, dist = get_matching(newobj,detected_objs[labelmap[i - 1]], tresh) # Return 0 if not in the list
                
                if num != 0: # object found --> estimate the spatial evidence
                    dt = newobj.get_time() - match_obj.get_time() 
                    prior_x, prior_y = predict_pos(match_obj,dt)
                    likelihood_x, likelihood_y  = newobj.get_p_upL()
                    x_distrib = update(prior_x, likelihood_x)
                    y_distrib = update(prior_y, likelihood_y)
                    
                    [x,y] = newobj.get_upL()
                    approx_spatial_diff = abs(x-x_distrib.mean)+abs(y-y_distrib.mean)  
                    # Experimental
                    #spatial_evidence = 1 - approx_spatial_diff/300
                    
                    if (dist+approx_spatial_diff/900) < tresh: # If spatial difference is small, it will keep
                        detected_objs[labelmap[i - 1]][match_obj.get_id()-1] = newobj.set_id(num) # Update with the matched newer object
                    else:
                        print('Distance of new object:',approx_spatial_diff)
                        num = len(detected_objs[labelmap[i-1]])+1 # Add as new image as well
                        detected_objs[labelmap[i-1]].append(newobj.set_id(num))
                    
                else: # Add if the same image was never seem before
                    num = len(detected_objs[labelmap[i-1]])+1
                    detected_objs[labelmap[i-1]].append(newobj.set_id(num))
                
                cv2.rectangle(frame,
                            (int(pt[0]), int(pt[1])), # UPPER LEFT
                            (int(pt[2]), int(pt[3])), # LOWER RIGHT
                            COLORS[i % 3], 2)
                score = detections[0,i,j,0]
                cv2.putText(frame, labelmap[i - 1]+'('+str(num)+'):'+str(int(100* score)), (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
            j += 1
            pts.append(pt)
            
    return frame, pts, detected_objs # We return the original frame with the detector rectangle and the label around the detected object.

# Select Image
#imnames = ['yale1.mpg','yale2.mpg','iceskating.mpg']
#imnames_out = ['out_yale1','out_yale2','out_iceskating']
#imnames_out = ['tracked_yale1','tracked_yale2','tracked_iceskating']
imnames = ['yale1.mpg','yale2.mpg']
imnames_out = ['tracked_yale1_july12','tracked_yale2_july12']

i = 0
use_cuda = True

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('weights/ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# Creating the transformation (STOPPED HERE!)
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.

# Doing some Object Detection on a video
if use_cuda:
    net = net.cuda()

# Tracking part
tresh = 0.4 # UPDATE
detected_objs = {} # store by label name
    
#reader = imageio.get_reader('funny_dog.mp4') # We open the video.
reader = imageio.get_reader(imnames[i])
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer(imnames_out[i]+'.mp4', fps = fps) # We create an output video with this same fps frequence.
for j, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame, pts, detected_objs = detect(frame, net.eval(), transform, j, detected_objs) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(j) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.