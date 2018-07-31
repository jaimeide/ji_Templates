# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:06:51 2018

@author: jaime
"""
import numpy as np
import cv2
# Define some useful tools from filterpy
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])

class myobj():
    def __init__(self, time, frame, num=0, label='??', pt=[1,1,10,10], vxy=[0,0]): # pt: [upL_x, upL_y, downR_x, downR_y]
        self.time = time
        self.num = num
        self.label = label
        #self.upL_x = gaussian(pt[0],5)
        #self.upL_y = gaussian(pt[1],5)
        #self.downR_x = gaussian(pt[2],5)
        #self.downR_y = gaussian(pt[3],5)
        self.x = gaussian((pt[2]+pt[0])/2,5) # Get the center of bbox
        self.y = gaussian((pt[3]+pt[1])/2,5)
        self.vx = gaussian(vxy[0],5)
        self.vy = gaussian(vxy[1],5)
        # Store masked image
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[pt[1]:pt[3],pt[0]:pt[2]] = 255
        masked_img = cv2.bitwise_and(frame,frame,mask = mask)
        self.img = masked_img 
    
    def set_id(self,num):
        self.num = num
        return self
    def set_vxy(self,vxy):
        self.vx = gaussian(vxy[0],5)
        self.vy = gaussian(vxy[1],5)
        return self
    
    def get_time(self):
        return self.time

    def get_label(self):
        return self.label
    
    def get_id(self):
        return self.num
    
    def get_img(self):
        return self.img
    
    #def get_p_upL(self):
    #    return self.upL_x, self.upL_y
    
    #def get_upL(self):
    #    return [self.upL_x.mean, self.upL_y.mean]
    
    #def get_downR(self):
    #    return [self.downR_x.mean, self.downR_y.mean]
    
    def get_p_xy(self):
        return self.x, self.y
    
    def get_xy(self):
        return [self.x.mean, self.y.mean]
    
    def get_vxy(self):
        return [self.vx.mean, self.vy.mean]
    
    
def get_bbox(pt,k_reduce=0.1):
    #k_reduce = 0.1
    deltaH = pt[3]-pt[1]
    deltaW = pt[2]-pt[0]
    ptx = [pt[0]+deltaW*k_reduce,pt[1]+deltaH*k_reduce, pt[2]-deltaW*k_reduce,pt[3]-deltaH*k_reduce]
    bbox = [int(x) if x>0 else 0 for x in ptx] # get positive and integer points 
    return bbox

def showme(img):
    '''
    # Testing mask
    mask = np.zeros_like(frame)  # init mask
    #mask = np.zeros(frame.shape[:2], np.uint8)
    pt = [100,200,300,400]
    mask[int(pt[0]):pt[2], pt[1]:pt[3],0] = 255
    mask[int(pt[0]):pt[2], pt[1]:pt[3],1] = 255
    mask[int(pt[0]):pt[2], pt[1]:pt[3],2] = 255
    
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[int(pt[0]):pt[2], pt[1]:pt[3]] = 255
    masked_img = cv2.bitwise_and(frame,frame,mask = mask)
    cv2.imshow('frame', masked_img)
    key = cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()
    '''
    while True:
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow('frame',img)
        if key == 27:
            break
    cv2.destroyAllWindows()
    
def showhist(img):
    # Analyze
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr[1:],color = col)
        plt.xlim([0,256])
    plt.show()

def compare_hist(img1,img2,method):
#    OPENCV_METHODS = (
#	("Correlation", cv2.cv.CV_COMP_CORREL),      # High, better
#	("Intersection", cv2.cv.CV_COMP_INTERSECT),  # High, better 
    #	("Chi-Squared", cv2.cv.CV_COMP_CHISQR),   # Low, better
#	("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA)) # Low, better   

#cv2.cv.CV_COMP_CORREL:: cv2.HISTCMP_CORREL
#cv2.cv.CV_COMP_CHISQR :: cv2.HISTCMP_CHISQR/ HISTCMP_CHISQR_ALT
#cv2.cv.CV_COMP_INTERSECT :: cv2.HISTCMP_INTERSECT
#cv2.cv.CV_COMP_BHATTACHARYYA :: cv2.HISTCMP_BHATTACHARYYA

    return cv2.compareHist(get_hist(img1),get_hist(img2), method)

def get_matching(newobj,objs,tresh):
    num = 0
    min_dist = 1
    match = []
    for obj in objs:
        #aux1 = newobj.get_img()
        #aux2 = obj.get_img()
        dist = compare_hist(newobj.get_img(),obj.get_img(),cv2.HISTCMP_BHATTACHARYYA)
        #showme(frame)
        #showme(obj.get_img())
        if dist< tresh and dist < min_dist: # keep the one with minimum distance
            num =  obj.get_id()
            match = obj
            min_dist = dist

            
    if num==0:
        print('-- NEW obj dist(hist): %.2f'%(dist),end='')
    else:
        print(' -- Obj found dist(hist): %.2f'%(dist),end='')
        
    return num, match, min_dist

def get_hist(img):
    # extract a 3D RGB color histogram from the image,
	# using 12 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([img], [0, 1, 2], None, [12, 12, 12],[0, 256, 0, 256, 0, 256])
	return cv2.normalize(hist,hist).flatten()


# Kalman filter tools (filterpy)
def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

def predict_pos(obj,dt):
    [x,y] = obj.get_xy() # Use upper left corner as the velocity reference.
    [vx,vy] = obj.get_vxy()
    #movx = gaussian(vx*dt, 5) # simplify...
    #movy = gaussian(vy*dt, 5)
    return gaussian(x + vx*dt, 10), gaussian(y + vy*dt, 10) 

