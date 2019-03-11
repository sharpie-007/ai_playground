# AI Playground

# Background and Description

## Problem Description

We hear about AI everywhere, and every time it gets mentioned, the reality, and fiction behind what AI is and could be gets a little more blurry. Terms can be confusing (Computer Vision, Object Detection, Natural Language Detection, Narrow AI, General AI, etc) and are often intermingled and mixed without consideration. This makes it very difficult for people who aren't in the field, but are required to be able to procure, implement, and ultimately move their businesses in the direction of Artificial Intelligence. How might we desmystify AI in an engaging, simple, and powerful way so that non Data Science profesisonals can grasp it's potential and it's limits?

## My Solution, the AI Playground.

I decided that using Jupyter Notebooks, while an enormous leap forward from showing people source code and or command line interfaces, wouldn't be an impactful way to let people interact with trained classifiers. I decided instead to build out a couple of demonstration capabilities that regular users could interact with. This way a user could experience some of the benefits, advantages, drawbacks, and shortcomings of some of applied machine learning. I broke this down into three demonstrations.

1. Static Image Object Detection (Using a pre-trained, open source model)
2. Video Based Object Detection (Using a pre-trained, open source model)
3. Binary Text Classification (including all source code to prepare and train a Keras Model)

I chose these three demonstrations because Computer Vision gets a lot of attention, and it's visually very compelling to interact with, and text classification because Natural Language Processing (NLP) is a passion of mine and I want to shine more light on the issues around cyber bullying and the potential things we can do to limit it using more advanced techniques. 

In each demonstration, the user will see some of the positives (detecting subversive speech, blurry objects) and negatives (unable to detect sarcasm, limited vocabulary, missing objects, misclassification) of using each model. Again, the intent is to demystify the process somewhat.

N.B., the video based object detection demo is non-interactive. Having a dynamic one (i.e. user uploaded videos), would have been too costly to deploy (both in compute and in monetary terms).

## Application Set up

The solution is deployed as a flask app, and requires two things to be completed before it will run:

1. You'll need to download the YOLOv3 weights for the coco model from Joseph Redmon's site here: https://pjreddie.com/media/files/yolov3.weights and drop it into the yolo-coco directory.
2. You'll want to run train_cyber.py to build your own cyber model.

Once both are complete you can run it in the app folder with `python run.py`

## YOLOv3 and Object Detection

You can download the weight file here: https://pjreddie.com/media/files/yolov3.weights. The overall Convolusional Neural Net (CNN) is based on the research from Joseph Redmon. https://pjreddie.com/darknet/yolo/. For the implementation of the static object classifier, I started with source code from pyimagesearch (https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/) and then heavily modified it to work with the web site, catch url's, and produce object summaries.

The object detection function takes a url from the user, tries to grab it with urllib, converts it to an array, resizes it, passes it through the CNN, then produces a summary of the ojbects it detected. It drops out a new file with the bounding boxes on it which is rendered side by side in the web page. I don't elaborate on the CNN construction as this is well covered here: https://pjreddie.com/darknet/yolo/


### Site Design.

The site uses bootstrap.js for it's design foundation, and the specific layout is based on the Product template (https://getbootstrap.com/docs/4.0/examples/product/)

