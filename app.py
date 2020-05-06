import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


# Instantiate Flask object
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Type of iris plant: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

