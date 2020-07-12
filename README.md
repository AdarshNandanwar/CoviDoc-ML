# CoviDoc-ML
Flask Backend server made using Keras and Tensorflow for accessing ML features of the CoviDoc App.  
Algorithm based on the paper:  
Determination of SpO2 and Heart-rate using Smartphone Camera, Kanva et al.  
https://www.iiitd.edu.in/noc/wp-content/uploads/2017/11/06959086.pdf
## Requirements
To run the server, you will need Flask, Keras, TensorFlow, matplotlib and some other packages.
## Installation
Clone this repository into your system
```bash
git clone https://github.com/AdarshNandanwar/CoviDoc-ML.git
```
## Usage
Create and activate a virtualenv.
```bash
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
virtualenv venv 
source venv/bin/activate
```
Install required dependencies.
```bash
pip install -r requirements.txt
```
Run the server.
```bash
python app.py
```
## API
**Supported Media Types**
```
multipart/form-data
```
#### POST http://127.0.0.1:5000/analyze-xray/
**Form Parameters**
```
file: xray image in jpeg, jpg orpng format
```
**Response**
```
{
    "data": {
        "prediction": 0
    },
    "message": "",
    "status": "success"
}
```
#### POST http://127.0.0.1:5000/analyze-video/
**Form Parameters**
```
file: video feed of finger with torch on in mp4 format
```
**Response**
```
{
    "data": {
        "bpm": 84,
        "bpm_disc": 84,
        "spo2": 98,
        "spo2_disc": "normal"
    },
    "message": "",
    "status": "success"
}
```
