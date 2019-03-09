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

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=True,
        help="file that contains the dataset")
    ap.add_argument("--vocab_size", type=float, default=500,
        help="size of vocabulary to fit the tokeniser on")
    ap.add_argument("--epochs", type=float, default=25,
        help="number of epochs for training the model")
    args = vars(ap.parse_args())

    vocab_size = int(args['vocab_size'])
    epochs = int(args['epochs'])


    print("\nLoading {} from disk...".format(args['filename']))
    training_data = pd.read_json(args['filename'], lines = True)
    print("Done!")

    print("\nConverting Labels...")

    labels = []
    for i in tqdm(range(0, len(training_data))):
        labels.append("".join(training_data.iloc[i]['annotation']['label']))
    training_data['label'] = labels
    
    print("\nFitting Tokenizer...")

    tokenizer, X = tokenize(training_data['content'], vocab_size)

    print("\nSplitting into Training and Test data...")

    y = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15, random_state=42)

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

    class_weight = {0: 1,
                1: 1.68}

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
    
    print("\n=============CLASSIFICATION RESULTS=============")
    print("\n================Confusion Matrix================")
    print(confusion_matrix(y_test[:,1], predictions))
    print("\n=============Classification Report==============")
    print(classification_report(y_test[:,1], predictions))

    # serialize model to JSON
    print("\nSaving trained model and tokenizer")

    model_json = model.to_json()
    with open("cyber_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cyber_model.h5")
    print("Saved model to disk")

    dump(tokenizer, 'cyber_tokenizer.joblib')
    dump(history, 'cyber_model_history.joblib')




    


    

