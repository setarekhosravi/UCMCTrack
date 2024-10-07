#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Mon Oct 7 2024 09:52:22
    Code for tracking multiple objects using UCMCTrack,
    and YOLOv5 as Detector. 
    Store results in the mot format.
    @author: STRH
"""

import torch
import os,cv2
import argparse

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np

# constant val
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

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
    def __init__(self, args):
        self.seq_length = 0
        self.gmc = None
        self.args = args

    def read_gt(self, file_path):
        # Read and parse the file
        with open(file_path, 'r') as file:
            gt_lines = file.readlines()
        
        # Initialize a dictionary to hold lists of boxes for each frame
        frames_boxes = {}
        
        # Process each line in the file
        for line in gt_lines:
            # Split the line by commas
            parts = line.strip().split(',')
            # Extract relevant fields
            frame_id = int(parts[0])
            object_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            class_id = int(parts[7])
            visibility = float(parts[8])
            
            # Create a bounding box tuple
            bbox = [x, y, x + width, y + height, 1, class_id]
            
            # Add the bounding box to the corresponding frame's list
            if frame_id not in frames_boxes:
                frames_boxes[frame_id] = []
            frames_boxes[frame_id].append(bbox)
        
        # Let's print the first few frames to see the result
        frames_boxes_sorted = dict(sorted(frames_boxes.items()))
        return frames_boxes_sorted

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        # selct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if not self.args.gt:
            # loading model using torch.hub
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path= self.args.weights, force_reload= False)
            self.model.float()
            self.model.eval()

        else:
            self.gt_path = self.args.gt

    def get_dets(self, img,conf_thresh = 0,det_classes = [0], frame_id=1):
        
        dets = []

        # Convert frames from BGR to RGB (because OpenCV uses BGR format)  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # Using YOLOv5 for inference 
        results = self.model(frame)
        preds = results.xyxy[0].cpu().numpy()
        # print(results)

        det_id = 0
        if not self.args.gt:
            for pred in preds:
                # conf = box.conf.cpu().numpy()[0]
                # bbox = box.xyxy.cpu().numpy()[0]
                # cls_id  = box.cls.cpu().numpy()[0]
                x1,y1,x2,y2, conf, cls_id = pred
                w = x2 - x1
                h = y2 - y1
                if cls_id not in det_classes or conf <= conf_thresh:
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

        else:
            preds = self.read_gt(self.gt_path)[frame_id]
            for pred in preds:
                # conf = box.conf.cpu().numpy()[0]
                # bbox = box.xyxy.cpu().numpy()[0]
                # cls_id  = box.cls.cpu().numpy()[0]
                x1,y1,x2,y2, conf, cls_id = pred
                w = x2 - x1
                h = y2 - y1
                if cls_id not in det_classes or conf <= conf_thresh:
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
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", width, height)

    detector = Detector(args=args)
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score,False,None)

    # Loop to read video frames
    frame_id = 0
    while True:
        ret, frame_img = cap.read()
        if not ret:  
            break
    
        dets = detector.get_dets(frame_img,args.conf_thresh,class_list, frame_id=frame_id)
        # print(dets)
        tracker.update(dets,frame_id + 1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--video', type=str, default = "/home/setare/Vision/Work/test/S500 'Night Flyer' Quadcopter agility testing.mp4", help='video file name')
    parser.add_argument('--cam_para', type=str, default = "demo/cam_para_test1.txt", help='camera parameter file name')
    parser.add_argument('--weights', type=str, help="detector weights path")
    parser.add_argument('--wx', type=float, default=5, help='wx')
    parser.add_argument('--wy', type=float, default=5, help='wy')
    parser.add_argument('--vmax', type=float, default=10, help='vmax')
    parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
    parser.add_argument('--input_type', type=str, required=True, help="Input type: image or video")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input images folder or video file")
    parser.add_argument('--save_mot', type=str, required=True, help="save results in mot format")
    parser.add_argument('--save_path', type=str, required=False, help="path to folder for saving results")
    parser.add_argument('--gt', type=str, required=False, help="path to gt.txt file")
    parser.add_argument('--save_video', type=str, required=False, help="if you want to save the tracking result visualization set it True")

    args = parser.parse_args()

    main(args)



