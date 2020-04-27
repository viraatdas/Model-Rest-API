# Load libraries
import flask
import keras
import tensorflow as tf
from keras.models import load_model
import pandas as pd

# instantiate flask 
app = flask.Flask(__name__)

# load the model
model = load_model('games.h5')

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        data["prediction"] = str(model.predict(x))
        data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0', threaded=False)