# import panda, keras and tensorflow

def train_model():
    import pandas as pd
    import tensorflow as tf
    import keras
    from keras import models, layers
    import numpy as np
    # Load the sample data set and split into x and y data frames 
    df = pd.read_csv("../train_data/initial_train.csv")
    import os
    os.remove("../train_data/initial_train.csv")

    x = df.drop(['label'], axis=1)
    y = df['label']
    # Define the keras model
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10,)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile and fit the model
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',
                metrics=['mse'])
    history = model.fit(x, y, epochs=100, batch_size=100,
                        validation_split = .2, verbose=0)


    # Save the model in h5 format 
    model.save("games.h5")

def predict():
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect('sqlite_default')

    model = model.load_model("games.h5")
    df = pd.read_csv("../test_data/test.csv")

    x = df.drop(['label'], axis=1)

    df = model.predict(x)
    print(df)
    df.to_sql('predictions', conn, if_exists='replace', index=False)


