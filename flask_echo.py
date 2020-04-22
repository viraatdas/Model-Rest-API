# load Flask
import flask
app = flask.Flask(__name__)

# predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    # get request parameters
    params = flask.request.json

    # if params not found
    if params == None:
        params = flask.request.args

    # if params found
    if params != None:
        data["response"] = params.get("msg")
        data["success"] = True

    # return response in json format
    return flask.jsonify(data)

# start flask app, allow remote connections
app.run(host='0.0.0.0')
