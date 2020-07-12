from flask import Flask, render_template, url_for, request, redirect, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from video_processing import validate_video, process_video, calculate_heart_rate, calculate_spo2
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__) 

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4"]

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def allowed_video(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return True
    else:
        return False
  
# ROUTES

@app.route('/', methods = ['GET'])
def index():
    return "<h3>POST API:</h3>for xray analysis - /analyze-xray/<br>for video analysis - /analyze-video/"

@app.route('/analyze-xray', methods = ['POST', 'GET'])
def analyze_xray():
    if request.method == 'POST' and len(request.files['file'].filename) and allowed_image(request.files['file'].filename):
        res = {
            "status": "success",
            "data": {
                "prediction": None
            },
            "message": ""
        }
        target = os.path.join(APP_ROOT, 'xray_images')
        image_path = os.path.join(target, "default.jpeg")
        try:
            file = request.files['file']
            filename = secure_filename(file.filename)
            image_path = os.path.join(target, filename)
            file.save(image_path)
            model = load_model(os.path.join(APP_ROOT, 'ml_model', 'xray_model.h5'))
            img = image.load_img(image_path,target_size=(224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img,axis=0)
            res['data']['prediction'] = int(model.predict_classes(img)[0][0])
        except Exception as e:
            res['status'] = "error"
            res['message'] = str(e)
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify(res)

    else:
        return "Send POST request here for xray classification"

@app.route('/analyze-video', methods = ['POST', 'GET'])
def analyze_video():
    if request.method == 'POST' and len(request.files['file'].filename) and allowed_video(request.files['file'].filename):
        res = {
            "status": "success",
            "data": {
                "spo2": None,
                "spo2_disc": None,
                "bpm": None,
                "bpm_disc": None
            },
            "message": ""
        }
        target = os.path.join(APP_ROOT, 'finger_videos')
        video_path = os.path.join(target, 'default.mp4')
        try:
            file = request.files['file']
            filename = secure_filename(file.filename)
            video_path = os.path.join(target, filename)
            file.save(video_path)
            video = cv2.VideoCapture(video_path)
            validate_video(video)
            video = process_video(video)

            res['data']['spo2'] = calculate_spo2(video)
            res['data']['spo2_disc'] = calculate_spo2(video, discretize=True)
            res['data']['bpm'] = calculate_heart_rate(video)
            res['data']['bpm_disc'] = calculate_heart_rate(video, discretize=True)

        except Exception as e:
            res['status'] = "error"
            res['message'] = str(e)
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify(res)

    else:
        return "Send POST request here for video prediction"
  
if __name__ == '__main__': 
   app.run(port = 4200)