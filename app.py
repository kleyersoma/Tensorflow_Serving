# In[] Load dependencies
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import requests
import base64
import json
from io  import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import cv2

# In[] TensorFlow Serving Assets
HEADERS = {'content-type':'application/json'}
MODEL_API_URL = 'http://localhost:8501/v1/models/fashionmnist_model_serving/versions/1:predict'
CLASS_NAMES = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

# In[] Instantiate Flask App
app = Flask(__name__)
CORS(app)

# In[] Image resizing utils
def resize_image_array(img, img_size_dims):
    img = cv2.resize(img,
                     dsize=img_size_dims,
                     interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    return img

# In[] Model warmup function
def warmup_model_serve(warmup_data, warmup_labels, img_dims=(32, 32)):
    warmup_data_processed = (np.array([resize_image_array(img, 
                                                          img_size_dims=img_dims) 
                                            for img in np.stack([warmup_data]*3, 
                                                                axis=-1)])) / 255.
    data = json.dumps({"signature_name": "serving_default", 
                       "instances": warmup_data_processed.tolist()})

    json_response = requests.post(MODEL_API_URL, data=data, headers=HEADERS)
    predictions = json.loads(json_response.text)['predictions']
    print('Model warmup complete') # log this in actual production
    predictions = np.argmax(np.array(predictions), axis=1)
    print(classification_report(warmup_labels, predictions))

# In[] TensorFlow Serving lazy loads so the model can be warmed up with sample data
# This runs as soon as the web service is set up to run
warmup_data = np.load('serve_warmup_data.npy')
warmup_labels = np.load('serve_warmup_labels.npy')
warmup_model_serve(warmup_data, warmup_labels)

# In[] Liveness Test
@app.route('/apparel_classifier/api/v1/liveness', methods=['GET', 'POST'])
def liveness():
    return 'API Live!'

# In[] Model Inference Endpoint
@app.route('/apparel_classifier/api/v1/model_predict', methods=['POST'])
def image_classifier():
    img = np.array([keras.preprocessing.image.img_to_array(
            keras.preprocessing.image.load_img(BytesIO(base64.b64decode(request.form['b64_img'])),
                                               target_size=(32, 32))) / 255.])

    data = json.dumps({"signature_name": "serving_default", 
                       "instances": img.tolist()})
    
    json_response = requests.post(MODEL_API_URL, data=data, headers=HEADERS)
    prediction = json.loads(json_response.text)['predictions']
    prediction = np.argmax(np.array(prediction), axis=1)[0]
    prediction = CLASS_NAMES[prediction]

    return jsonify({'apparel_type': prediction})

# In[] Running REST interface, port=5000 for direct test use debug=True when debugging
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
