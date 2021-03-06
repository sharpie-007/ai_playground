# AI Playground

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
e.g. `python train_cyber.py -f <filename> --vocab_size 500 --epochs 50`

Once both are complete you can run it in the app folder with `python run.py`

## YOLOv3 and Object Detection

You can download the weight file here: https://pjreddie.com/media/files/yolov3.weights. The overall Convolusional Neural Net (CNN) is based on the research from Joseph Redmon. https://pjreddie.com/darknet/yolo/. For the implementation of the static object classifier, I started with source code from pyimagesearch (https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/) and then heavily modified it to work with the web site, catch url's, and produce object summaries.

The object detection function takes a url from the user, tries to grab it with urllib, converts it to an array, resizes it, passes it through the CNN, then produces a summary of the ojbects it detected. It drops out a new file with the bounding boxes on it which is rendered side by side in the web page. I don't elaborate on the CNN construction as this is well covered here: https://pjreddie.com/darknet/yolo/.

## The Cyber Aggression Classifier

train_cyber.py is designed to be a command line tool to train your own binary text classifier. It sources the dataset from https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls. You can choose the source file, configure the vocabulary size, and choose the number of epochs when creating the neural network. It uses the keras tokenizer function to convert the documents into an array, and then feeds it to the Neural Network. The outputs of the trained Neural Network, including the training history and classification report, are pushed out to files that are then picked up via the web app and displayed on the NLP page. This way the user can see how the model was trained, it's accuracy, convergence, etc. 

### Neural Network Design.

There's a lot of experimental design in the construction of the NN. By using the Tokenizer I was able to pass vectors to the NN instead of something more simple like BoW. The option I used here was matrix, as the complexity of the overall texts was very low. (they're tweets). I may add the ability to change this via the CLI tool at a later stage. You can read more on the Keras Tokenizer function here: https://keras.io/preprocessing/text/.

### Layers

The network is built as follows:

`model = Sequential()` <br/>
`model.add(Embedding(250, 8, input_length=vocab_size))`<br/>
`model.add(Dropout(0.3))`<br/>
`model.add(Flatten())`<br/>
`model.add(Dense(750, activation='relu'))`<br/>
`model.add(Dropout(0.2))`<br/>
`model.add(Dense(75, activation='relu'))`<br/>
`model.add(Dense(2, activation='softmax'))`<br/>

You can see that we take in a 3 dimensional matrix, then pass it to a 250 neuron layer, then drop 30% of the neurons. Then we flatten it, create a dense layer, dropout 20% of that layer, then add another smaller dense layer, then the output layer. We add the dropout layers to reduce the probability of overfitting the data on the training side, although if you train the model too long (too many epochs) you can still eventually overfit it. I experimented with the design of the network a lot, and this one eventually hit the right amount of compute, convergence, and accuracy for me. I use relu as the activation for the hidden layers and softmax for the output layer so that we get a % probability on the output layer. Read more on softmax here: https://keras.io/activations/#softmax.

The overall outcomes are printed to the console in a classification report and then saved to disk to be picked up by the web app.

The web app displays the outcomes in tabular and chart form (I built the chart in plotly.js)

### Site Design.

The site uses bootstrap.js for it's design foundation, and the specific layout is based on the Product template (https://getbootstrap.com/docs/4.0/examples/product/)

