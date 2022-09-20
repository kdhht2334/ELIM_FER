#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:57:51 2018

@author: hwauni
"""
import sys

import cv2
import pickle
import numpy as np
import time
from random import shuffle
#from scipy.misc import imread, imresize
from imageio import imread
from timeit import default_timer as timer
from multiprocessing import Process, Lock, Queue
import datetime

class VideoMgr():    
    def __init__(self, camIdx, camName):        
        self.camIdx = camIdx
        self.camName = camName
        self.camCtx = None
        self.start = None
        self.end = None
        self.numFrames = 0

    def open(self, config):        
        self.camCtx = cv2.VideoCapture(self.camIdx)
        if not self.camCtx.isOpened():
            print('isOpend Invalid')
            print(self.camIdx)
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
            
        self.camCtx.set(cv2.CAP_PROP_FRAME_WIDTH, int(config['width']))
        self.camCtx.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config['height']))        
        self.camCtx.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(config['invformat'])))       
        self.camCtx.set(cv2.CAP_PROP_FPS, int(config['fps']))
        #self.camCtx.set(cv2.CAP_PROP_EXPOSURE, -1)

        #time.sleep(2) # give time for camera to init.
    
        # 'prime' the capture context...
        # some webcams might not init fully until a capture
        # is done.  so we do a capture here to force device to be ready
        #self.camCtx.read()
        
    def read(self):
        return self.camCtx.read()               

    def start(self):
        # start the timer
        self.start = datetime.datetime.now()
        return self
 
    def stop(self):
        # stop the timer
        self.end = datetime.datetime.now()
 
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self.numFrames += 1
 
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self.end - self.start).total_seconds()
         
    def fps(self):
        # compute the (approximate) frames per second
        return self.numFrames / self.elapsed()

    '''
    def run(self, config, cam_url, queue, is_ready):        
        cam = cv2.VideoCapture(cam_url)        
        if not cam.isOpened():
            print('isOpend Invalid')
            print(cam_url)
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(config['width']))
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config['height']))        
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(config['invformat'])))       
        #cam.set(cv2.CAP_PROP_EXPOSURE, -7.5)
        
        # Compute aspect ratio of video
        camWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camAr = camWidth/cam_height
        camFps = float(cam.get(cv2.CAP_PROP_FPS))
        camFormat = cam.get(cv2.CAP_PROP_FOURCC)
        camExposure = cam.get(cv2.CAP_PROP_EXPOSURE)
        
        print("[VM] Camera Info(id=%s" % self.cam_id, "name=%s" % self.cam_name, "width=%f" % camWidth, 
              "height=%f" % cam_height, "fps=%f" % camFps, "exposure=%f" % camExposure, "format=%f)" % camFormat)    
    
        curtime_str = time.strftime('%H%M%S')
        if config['videosave'] == 'on':                           
            outvfilename = config['outvpath'] + curtime_str + '_' + config['outvfilename']
            fourcc = cv2.VideoWriter_fourcc(*config['outvformat'])
            out = cv2.VideoWriter(outvfilename, fourcc, camFps, (camWidth, cam_height))                 
        
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        count = 0
        
        while(True):
            ret, orig_image = cam.read()       
               
            if ret:                
                # Calculate FPS
                # This computes FPS for everything, not just the model's execution 
                # which may or may not be what you want
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0                
                
                if is_ready.value:
                    #orig_image = cv2.resize(orig_image, (640,360))                    
                    queue.put((self.cam_id, self.cam_name, orig_image))
                
                if config['camshow'] == 'on':
                    cv2.putText(orig_image, fps, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                    cv2.imshow(self.cam_name, orig_image)                                          
                if config['videosave'] == 'on':                    
                    out.write(orig_image)
                if config['imagesave'] == 'on':
                    if count == 0 or curr_fps % int(config['outirate']) == 0:
                        outifilename = config['outipath'] + curtime_str + '_' + str(count) + '_' + config['outifilename']
                        cv2.imwrite(outifilename, orig_image)                
                        count = count + 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break             
        if config['videosave'] == 'on':          
            out.release()
        cv2.destroyAllWindows()
        cam.release()
    '''
    def close(self):
        self.camCtx.release()
    
    def reset(self):
        self.close()
        self.open()
