# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:12:28 2019

@author: jaime
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['interactive'] == True # will show figure with matplotlib

def show_plt(img):
    # Will expect RGB
    plt.imshow(img)
    plt.title('Look weird if you entered BGR...')
    plt.show()
    

def show_cv2(img):
    # Input should be BGR
    cv2.imshow('image',img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def split_img(img,nhorizontal,nvertical,n):
    # Function will split and return the image counting from Left to Down
    # n starting from "0"
    [height,width,_] = img.shape
    xsize = width // nhorizontal
    ysize = height // nvertical
    y = n // nhorizontal  # quocient
    x = n % nhorizontal # remainder
    print('ysize:',ysize)
    print('xsize:',xsize)
    print('y:',y)
    print('x:',x)
    return img[y*ysize:(y+1)*ysize, x*xsize:(x+1)*xsize, :] 

def main():
    fname = 'carparking.jpg'
    img = cv2.imread(fname) # BGR
    
    img1 = split_img(img,3,2,1)
    # Use opencv to convert color scheme: RGB --> 
    img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    plt.imshow(img2)
    plt.show()
    #show_plt(img)
    #show_cv2(img)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()