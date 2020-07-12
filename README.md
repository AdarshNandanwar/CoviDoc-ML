# CoviDoc-ML
Flask Backend server made using Keras and Tensorflow for accessing ML features of hte CoviDoc App.
## Requirements
To run the server, you will need Flask, Keras, TensorFlow, matplotlib and some other packages.
### Installation
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
POST http://127.0.0.1:5000/analyze-xray/
POST http://127.0.0.1:5000/analyze-video/
## Deployed Server
Currently not deployed