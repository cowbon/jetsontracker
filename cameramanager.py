#!/usr/bin/env python

from __future__ import print_function
import cv2
import json

DEFAULT_HEIGHT = 
DEFAULT_WIDTH = 
onboard = 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,\
format=(string)I420, framerate=(fraction)30/1  ! nvvidconv flip-method=2 ! video/x-raw,\
format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

rtsp_suffix = ' ! latency = 0 ! decodebin ! videoconvert ! appsink'

class CameraManager():
    def __init__(self, config_path):
      	with open(config_path, 'r') as f:
            config = json.load(config_path)
            self.sdp, self.video = [], []   
            if config['sdp']:
                for conn in config['sdp']:
                    self.sdp.append(conn['link'])

            if config['video']:
                for conn in config['video']:
                    self.video.append(conn['path'])

            if device_count() == 0:
                self.onboard = True


    def device_count(self):
        return len(self.sdp) + len(self.video)


    def get_device(self, device=0):
        def build_str(device):
            if device == 'onboard':
                return onboard

            else:
                return 'rtspsrc location=' + device + rtsp_suffix

        def _get_divice(self, device):
	    if device < len(self.sdp)
	        return cv2.VideoCapture(build_str(self.sdp[device]), cv2.GSTREAMER)
	    else
	        # Get video from file
	        return cv2.VideoCapture(self.video[device-len(self.sdp)])

        if self.onboard = True:
            return cv.VideoCapture(build_str('onboard'), cv2.GSTREAMER)
        else:
            try:
                if type(device) is int:
                    return _get_device(device)

                elif type(device) is list:
                    return_list = []
                    for _device in device:
                        return_list.append(_get_device(_device))

                    return return_list

            except TypeError:
                print('Device must ba a integer or a list')
