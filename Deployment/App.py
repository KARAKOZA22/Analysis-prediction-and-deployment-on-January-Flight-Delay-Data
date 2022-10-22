import numpy as np
import pickle
from flask import Flask, request, jsonify,render_template
from sklearn import *
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=["POST"])
def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        return render_template('index.html', prediction_text='The flight will not be delayed')
    else:
        return render_template('index.html', prediction_text='The flight will be delayed')
    
if __name__ == "__main__":
    app.run(debug=True)
