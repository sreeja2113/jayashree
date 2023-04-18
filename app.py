from flask import Flask, request,jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS,cross_origin
import base64
# from flask_ngrok import run_with_ngrok

from pymongo import MongoClient
app = Flask(__name__)
# run_with_ngrok(app)
CORS(app, origins=["http://localhost:3000"])
model = load_model("sudeep.h5")
mongo_client=MongoClient("mongodb://hellokcr:Konepalli@ac-ts0ftlx-shard-00-00.hpocola.mongodb.net:27017,ac-ts0ftlx-shard-00-01.hpocola.mongodb.net:27017,ac-ts0ftlx-shard-00-02.hpocola.mongodb.net:27017/?ssl=true&replicaSet=atlas-zgec31-shard-0&authSource=admin&retryWrites=true&w=majority")

db = mongo_client['test']
collection = db['cellimages']


def preprocess_image(image):
    image = image.resize((50, 50))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
@app.route("/past-predictions", methods=["POST"])
def get_past_predictions():
  email=request.form["email"]

  result = collection.find({"email":email}, {"_id": 0, "image_b64": 1, "plabel": 1, "alabel": 1})
  return jsonify(list(result))


@app.route("/store-image", methods=["POST"])
def store_image():
  img_b64 = request.files["image_b64"].read()
  prediction = request.form["plabel"]
  truth = request.form["alabel"]
  mail=request.form["email"]
  db_result = collection.insert_one({
    'image': img_b64,
    'plabel': prediction,
    'alabel': truth,
    'image_b64': base64.b64encode(img_b64).decode('utf-8'),
    "email":mail
  })
  # email = mail
  print(db_result)
  return jsonify({"message": "Image stored successfully!"})
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    imagefile = request.files["imagefile"]
    image = Image.open(imagefile)
    image_array = preprocess_image(image)
    pred = model.predict(image_array)
    print(pred[0][0])
    if(pred[0][0]>=0.5):
      return "infected"
    else: 
       return "uninfected"    

if __name__ == "__main__":
    app.run(debug=True)
    # app.run()

