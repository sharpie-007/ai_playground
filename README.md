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

N.B., the video based object detection demo is non-interactive. Having a dynamic one (i.e. user uploaded videos), would have been too costly to deploy (both in computer and in monetary terms).

## A word on YOLOv3

You can download the weight file here: https://pjreddie.com/media/files/yolov3.weights. The overall Convolusional Neural Net (CNN) is based on the research from Joseph Redmon. https://pjreddie.com/darknet/yolo/.

