#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os,cv2
import argparse

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np


# Define a Detection class, including id, bb_left, bb_top, bb_width, bb_height, conf, det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class {}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector class, used to obtain the results of target detection from the Yolo detector
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        # selct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # loading model using torch.hub
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path= "/home/setare/Vision/yolov5/yolov5s.pt", force_reload= False)
        self.model.float()
        self.model.eval()

    def get_dets(self, img,conf_thresh = 0,det_classes = [0]):
        
        dets = []

        # Convert frames from BGR to RGB (because OpenCV uses BGR format)  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # Using YOLOv5 for inference 
        results = self.model(frame)
        preds = results.xyxy[0].cpu().numpy()
        # print(results)

        det_id = 0
        for pred in preds:
            # conf = box.conf.cpu().numpy()[0]
            # bbox = box.xyxy.cpu().numpy()[0]
            # cls_id  = box.cls.cpu().numpy()[0]
            x1,y1,x2,y2, conf, cls_id = pred
            w = x2 - x1
            h = y2 - y1
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # Create a new Detection object
            det = Detection(det_id)
            det.bb_left = x1
            det.bb_top = y1
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1

            dets.append(det)

        return dets
    

def main(args):

    class_list = [0]

    cap = cv2.VideoCapture(args.video)

    # Get fps of video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # Open a cv window and specify the height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score,False,None)

    # Loop to read video frames
    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret:  
            break
    
        dets = detector.get_dets(frame_img,args.conf_thresh,class_list)
        # print(dets)
        tracker.update(dets,frame_id)

        for det in dets:
            # Draw detection box
            if det.track_id > 0:
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left+det.bb_width), int(det.bb_top+det.bb_height)), (0, 255, 0), 2)
                # write the id of the detection box
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_id += 1


        # Show current frame
        cv2.imshow("Frame", frame_img)
        cv2.waitKey(1)

        video_out.write(frame_img)
    
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()



parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "/home/setare/Vision/Work/test/S500 'Night Flyer' Quadcopter agility testing.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default = "demo/cam_para_test1.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
args = parser.parse_args()

main(args)



