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
import tensorflow as tf
import random

# Initialize YOLOv3 model before the app functions, keeps users from having to wait.

# The YOLOv3 implementation was originally a command line script from a tutorial on
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/. I've
# heavily modified it to provide summary classification results that can be rendered
# into this flask app and included the ability for a user to paste their own image
# url in to see how YOLO-COCO works.

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

cyber_tokenizer = joblib.load("../models/cyber_tokenizer.joblib")
json_file = open("../models/cyber_model.json")
model_json = json_file.read()
json_file.close()
cyber_model = model_from_json(model_json)
cyber_model.load_weights("../models/cyber_model.h5")
cyber_model._make_predict_function()

# Loading model outputs

cyber_model_history = joblib.load("../models/cyber_model_history.joblib")
cyber_model_perf_summary = joblib.load("../models/performance_summary.joblib")
cyber_df = pd.read_csv("../models/final_dataset.csv")


# YOLO OBJECT DETECTION
def process_images_complete(url_list, net=net):
    '''
    Function takes in a url and a CNN (default configured for YOLOv3), then 
    attempts to grab the image on the other side using URLLIB, convert it to
    an array, and then feed it through the Darknet CNN. As objects are detected,
    a summary dictionary is created to produce a report to be fed to the website
    for the user to review, as well as a new image. 
    Capable of handling multiple URLS, however the site is set up for single 
    URL's currently.

    Args:
        url_list: list of image urls, must be absolute (i.e. no google:data/ 
        urls)
        net: (Optional). The Neural net to perform the object detection with.

    Returns:
        results: summary dictionary of objects found in image.
        ***dumps a new image wih a bounding box to static/images
    '''

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

        # Generate the summary dictionary

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
    '''
    Flask function that takes outputs from cyber_training.py and renders them into
    summary tables and a plotly graph.

    Args: None

    Returns: 
        graphJSON: A prepared plotly graph data object.
        cyber_model_history: The history of the accuracy, loss, validation 
                            accuracy, and validation loss of the model training.
        cyber_model_perf_summary: a Dictionary containing the conventional 
        accuracy, f-score, recall, and precision metrics from sklearn along with the 
        five_random_agros: five 



    '''

    xScale = np.linspace(0, 1, len(cyber_model_history.history['val_loss']))
    y0_Scale = cyber_model_history.history['val_loss']
    y1_Scale = cyber_model_history.history['val_acc']
    y2_Scale = cyber_model_history.history['loss']
    y3_Scale = cyber_model_history.history['acc']
    Validation_Loss = graph_objects.Scatter(x = xScale, y = y0_Scale, name="Validation Loss")
    Validation_Accuracy = graph_objects.Scatter(x = xScale, y = y1_Scale, name="Validation Accuracy")
    Loss = graph_objects.Scatter(x = xScale, y = y2_Scale, name="Loss")
    Accuracy = graph_objects.Scatter(x = xScale, y = y3_Scale, name="Accuracy")
    data = [Validation_Loss, Validation_Accuracy, Loss, Accuracy]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)    
    five_random_agros = cyber_df['content'][cyber_df['label'] == 1].iloc[0:5].tolist()
    five_random_chills = cyber_df['content'][cyber_df['label'] == 0].iloc[0:5].tolist()

    return render_template('cyber_bullying.html',
    graphJSON = graphJSON,
    cyber_model_history = cyber_model_history,
    cyber_model_perf_summary = cyber_model_perf_summary,
    five_random_agros = five_random_agros,
    five_random_chills = five_random_chills)
    


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
    query_token = cyber_tokenizer.texts_to_matrix([query])
    cyber_prediction  = cyber_model.predict_proba(query_token)

    return render_template(
        'aggression_results.html',
        query=query,
        prediction = cyber_prediction
        
        # classification_result=classification_results
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()