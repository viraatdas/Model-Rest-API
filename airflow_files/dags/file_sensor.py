from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.dummy_operator    import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator


# from gen_model import train_model
# from gen_model import predict

import datetime
import airflow

def train_model(ds, **kwargs):
    import pandas as pd
    import tensorflow as tf 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import models, layers
    import numpy as np
    # Load the sample data set and split into x and y data frames 
    df = pd.read_csv("/Users/owner/airflow/train_data/initial_train.csv")
    import os
    os.remove("/Users/owner/airflow/train_data/initial_train.csv")

    x = df.drop(['label'], axis=1)
    y = df['label']
    # Define the keras model
    model = Sequential()
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




# https://airflow.apache.org/code.html#airflow.models.BaseOperator
default_args = {
    "depends_on_past" : False,
    "start_date"      : airflow.utils.dates.days_ago( 1 ),
    "retries"         : 1,
    "retry_delay"     : datetime.timedelta( hours= 5 ),
}

with airflow.DAG( "file_sensor_test_v2", default_args= default_args, schedule_interval= "*/5 * * * *", ) as dag:
    sensor_task = FileSensor( task_id= "my_file_sensor_task", poke_interval= 5, filepath="/Users/owner/airflow/train_data")
    model_train = PythonOperator(dag=dag,
               task_id='retrain_model',
               provide_context=True,
               python_callable= train_model)
    # predict_task = PythonOperator(dag=dag,
    #            task_id='predict_model',
    #            provide_context=False,
    #            python_callable= predict)
    start_task  = DummyOperator(  task_id= "start" )
    stop_task   = DummyOperator(  task_id= "stop"  )
    


start_task >> sensor_task >> model_train >> stop_task


# >> predict_task