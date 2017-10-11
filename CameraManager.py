#!/usr/bin/env python

from __future__ import print_function
import cv2
import json

DEFAULT_HEIGHT = 640
DEFAULT_WIDTH = 480
onboard = 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,\
format=(string)I420, framerate=(fraction)30/1  ! nvvidconv flip-method=2 ! video/x-raw,\
format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

rtsp_suffix = ' latency = 0 ! decodebin ! videoconvert ! appsink'


class CameraManager():
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.sdp, self.video = [], []
            if 'sdp' in config:
                for conn in config['sdp']:
                    print(conn)
                    self.sdp.append(conn['link'])

            if 'video' in config:
                for conn in config['video']:
                    self.video.append(conn['path'])

            self.onboard = True if self.device_count() == 0 else False

    def device_count(self):
        return len(self.sdp) + len(self.video)

    def get_device(self, device=0):
        def build_str(device):
            if device == 'onboard':
                return onboard

            else:
                return 'rtspsrc location=rtsp://' + device + rtsp_suffix

        def _get_device(self, device):
            if device < len(self.sdp):
		return build_str(self.sdp[device])
            else:
                # Get video from file
                return self.video[device-len(self.sdp)]

        if self.onboard:
            return build_str('onboard')
        else:
	    print(type(device))
            if type(device) is int:
                return _get_device(self, device)

            elif type(device) is list:
                return_list = []
                for _device in device:
                    return_list.append(_get_device(_device))

                return return_list

            else:
                print('Device must ba a integer or a list')
