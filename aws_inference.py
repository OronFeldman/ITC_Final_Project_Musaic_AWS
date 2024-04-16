from tensorflow.keras.models import load_model
from flask import Flask
from flask import request
import numpy as np
import json


app = Flask(__name__)


@app.route('/predict_album', methods=['POST'])
def inference_func():
    image_data = request.files['image'].read()
    img = np.frombuffer(image_data, dtype=np.float32).reshape((1, 299, 299, 3))
    classes_dict = json.loads(request.files['classes_dict'].read())
    names_dict = json.loads(request.files['names_dict'].read())
    predictions = model.predict(img)
    predicted_label = classes_dict[str(np.argmax(predictions, axis=1)[0])]
    album_name = names_dict[str(predicted_label)]
    return album_name


model = load_model('InceptionResNetV2_model')
app.run(host='0.0.0.0', port=8080)