import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tqdm import tqdm
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from joblib import dump

def tokenize(text, vocab_size):
    '''Tokenize reads in a list of strings and desired vocabulary size, fits the
    Keras tokenizer on the strings, encodes the strings and returns the fitted 
    tokenizer and encoded strings

    Args:
        text: list of strings to be fitted and encoded
        vocab_size: size of the vocabulary to build.

    Returns:
        encoded_docs: encoded array of strings 
        tokenizer: fitted tokenizer
    '''

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text)
    encoded_docs = tokenizer.texts_to_matrix(text, mode = 'count')
    return tokenizer, encoded_docs



if __name__ == "__main__":

    # Command line arguments capture
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=True,
        help="file that contains the dataset")
    ap.add_argument("--vocab_size", type=float, default=500,
        help="size of vocabulary to fit the tokeniser on")
    ap.add_argument("--epochs", type=float, default=25,
        help="number of epochs for training the model")
    args = vars(ap.parse_args())


    # Casting args as new variables for readability
    vocab_size = int(args['vocab_size'])
    epochs = int(args['epochs'])


    print("\nLoading {} from disk...".format(args['filename']))
    training_data = pd.read_json(args['filename'], lines = True)
    print("Done!")

    # Converting dict labels into columnar format
    print("\nConverting Labels...")

    labels = []
    for i in tqdm(range(0, len(training_data))):
        labels.append("".join(training_data.iloc[i]['annotation']['label']))
    training_data['label'] = labels
    
    print("\nFitting Tokenizer...")

    tokenizer, X = tokenize(training_data['content'], vocab_size)

    # Splitting test and train. The small sample size led me to use a small testing set.

    print("\nSplitting into Training and Test data...")

    y = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15, random_state=42)

    # Build the NN. See readme.md for the details on why this architecture

    print("\nCreating the Neural Network")
    model = Sequential()
    model.add(Embedding(250, 8, input_length=vocab_size))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(750, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    print(model.summary())

    # The dataset is imbalanced (see the web app) so boosting one class to
    # try to compensate.

    class_weight = {0: 1,
                1: 1.85}

    print("\nTraining Model")

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=256,
                        verbose=1,
                        validation_split=0.2,
                        class_weight=class_weight
                        )

    predictions = model.predict_classes(X_test, verbose = 1)
    full_predictions = model.predict_classes(X, verbose = 1)
    training_data['Predictions'] = full_predictions
    training_data.to_csv("final_dataset.csv")
    
    print("\n=============CLASSIFICATION RESULTS=============")
    print("\n================Confusion Matrix================")
    print(confusion_matrix(y_test[:,1], predictions))
    print("\n=============Classification Report==============")
    print(classification_report(y_test[:,1], predictions))


    # Create the dictionaries to be used by the web app

    confusion_matrix = (confusion_matrix(y_test[:, 1], predictions))
    performance_summary_dict = {
        "Data Characteristics":{
            "Total Phrases": training_data.shape[0],
            "Aggressive Phrases": len(training_data[training_data['label'] == "1"]),
            "Non Aggressive Phrases": len(training_data[training_data['label'] == "0"]),
            "Average Word Count": int(training_data['content'].apply(lambda x: len(x.split(" "))).mean()),
            "Shortest Phrase Length": training_data['content'].apply(lambda x: len(x.split(" "))).sort_values().iloc[0],
            "Longest Phrase Length": training_data['content'].apply(lambda x: len(x.split(" "))).sort_values(ascending = False).iloc[0]

            },
        "Confusion Matrix": {
            "True Positives": confusion_matrix[0][0],
            "False Positives": confusion_matrix[0][1],
            "False Negatives": confusion_matrix[1][0],
            "True Negatives": confusion_matrix[1][1]
            },
        "Classification Report":{
            "Precision": precision_score(y_test[:,1], predictions),
            "Recall": recall_score(y_test[:,1], predictions),
            "Accuracy": accuracy_score(y_test[:,1], predictions),
            "F1 Score": f1_score(y_test[:,1], predictions)
            }}
    

    # Save everything to disk

    # serialize model to JSON

    print("\nSaving trained model and tokenizer")
    model_json = model.to_json()
    with open("cyber_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5

    model.save_weights("cyber_model.h5")
    print("Saved model to disk")
    dump(tokenizer, 'cyber_tokenizer.joblib')
    dump(history.history, 'cyber_model_history.joblib')
    dump(performance_summary_dict, 'performance_summary.joblib')
    





    


    

