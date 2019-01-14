#!/usr/bin/env python3
from imutils.video import VideoStream
import numpy as np
import dropbox
import json
import argparse
import imutils
import datetime
import time
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-cl", "--classes", required=True, nargs='*',
                help="table of warning classes")
args = vars(ap.parse_args())

conf = json.load(open("/home/pi/Desktop/pi-object-detection/conf.json"))

client = dropbox.Dropbox(conf["dropbox_token"])
print("[SUCCESS] dropbox account linked")

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# initialize list of classes chosen to be suspicious
WARNING = args["classes"]
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt",
                               "MobileNetSSD_deploy.caffemodel")

# initialize the video stream 
vs = VideoStream(src=0).start()
time.sleep(2.0)

# path to folder for photos
path = conf["path_images"]
detection_count = 0

while True:
        actual_count = detection_count
	# grab the frame from the stream and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
	
        timestamp = datetime.datetime.now()
        

	# loop over the detections
        for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

		# filter out weak detections
                if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`
                        idx = int(detections[0, 0, i, 1])

                        if CLASSES[idx] in WARNING:
                            print("Detected " + CLASSES[idx] + " count: " + str(actual_count))
                            if actual_count == detection_count:
                                actual_count += 1

                            if actual_count > conf["min_frames"]:
                                ts = timestamp.strftime("%d%B%Y%I:%M:%S%p")
                                print("Warning! Detected " + CLASSES[idx] + "! Saving photo")
                                write_name = str(CLASSES[idx] + ts) + '.jpg'
                                abs_path = os.path.join(os.getcwd(), path, write_name)
                                cv2.imwrite(abs_path, frame)
                                path_drop = "/{base_path}/{name}".format(base_path=conf["dropbox_path"], name = write_name)
                                client.files_upload(open(abs_path, "rb").read(), path_drop)
                                detection_count = 0
                                actual_count = 0
                             
	# show the output frame
        if actual_count == detection_count:
            detection_count = 0
            actual_count = 0
        detection_count = actual_count
            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()