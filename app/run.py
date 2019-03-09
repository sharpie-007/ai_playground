import json
import plotly
import pandas as pd
import numpy as np
import cv2
import os
import time
import urllib
from flask import Flask
from flask import render_template, request, jsonify, url_for
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from keras.models import model_from_json
import joblib
import plotly.plotly as py
import plotly.graph_objs as graph_objects


# Initialize YOLOv3 model before the functions, keeps users from having to wait.

# YOLO CONFIGURATION
yolo_path = 'yolo-coco'
default_confidence = 0.5
default_threshold = 0.3

# Load YOLOv3 COCO Classes
labelsPath = os.path.sep.join([yolo_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Create Bounding Box Colours
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# Collect Weights and Config for Darknet-COCO
weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

# initialize and load the model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading Aggression Detection Model


def process_images_complete(url_list, net=net):

    output_dir = 'static'
    results = {}
    for i in range(0, len(url_list)):
        image_name = str(int(time.time())) + "_" + str(i) + ".png"
        start_grab = time.time()
        url = url_list[i]
        req = urllib.request.urlopen(url)
        end_grab = time.time()
        start_conversion = time.time()
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)  # 'Load it as it is'
        (H, W) = image.shape[:2]
        end_conversion = time.time()
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        download_time = end_grab - start_grab
        conversion_time = end_conversion - start_conversion
        image_height = H
        image_width = W
        processing_time = end - start
        print("Yolo took {:.4f} seconds to complete".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        default_confidence = 0.5
        default_threshold = 0.3
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > default_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, default_confidence, default_threshold)
        object_labels = []
        object_confidences = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                object_label = (LABELS[classIDs[i]])
                object_confidence = confidences[i]

                object_labels.append(object_label)
                object_confidences.append(object_confidence)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        cv2.imwrite(output_dir +"/" + image_name, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        object_summary = {}
        unique_objects = set(object_labels)
        for detected_object in unique_objects:
            object_summary[detected_object] = object_labels.count(detected_object)

        image_results = {"Image Height": image_height,
                   "Image Width": image_width,
                   "Download Time": download_time,
                   "Conversion Time": conversion_time,
                   "Object Detection Time": processing_time,
                   "Objects Detected": object_labels,
                   "Object Confidences": object_confidences,
                   "Object Summary": object_summary}
        
        results[image_name] = image_results

    return results


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/object_detection')
def object_detection():
    return render_template('object_detection.html')

@app.route('/object_detection_video')
def object_detection_video():
    return render_template('object_detection_video.html')

@app.route('/cyber_bullying')
def cyber_bullying():
    count = 500
    xScale = np.linspace(0, 100, count)
    yScale = np.random.randn(count)
    trace = graph_objects.Scatter(x = xScale, y = yScale)
    data = [trace]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)    
    return render_template('cyber_bullying.html', graphJSON = graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    query = request.args.get('query', '')
    print(query)
    results = process_images_complete([query])
    filename = next(iter(results))
    print(filename)
    summary = results[filename]['Object Summary']
    return render_template(
        'go.html',
        query=query,
        filename=filename,
        summary=summary
        # classification_result=classification_results
    )
@app.route('/detect_aggression')
def detect_aggression():
    query = request.args.get('query', '')
    print(query)

    return render_template(
        'aggression_results.html',
        query=query
        
        # classification_result=classification_results
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()