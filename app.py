import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
import torch
import mysql.connector
import threading
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, session,redirect,url_for,flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from datetime import timedelta
app = Flask(__name__)
app.secret_key = 'super-secret-key'
# Ensure the session cookie is temporary and doesn't last after the tab is closed
app.config['SESSION_COOKIE_PERMANENT'] = False  # Non-permanent session cookie
# Paths
UPLOAD_FOLDER = 'static/uploads'
PREDICT_FOLDER = 'static/predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

# MySQL config
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Use your own password if needed
    database="pothole"
)
cursor = mydb.cursor()

# Load models
resnet_model = load_model('models/resnet50.h5')
vgg_model = load_model('models/vgg19.h5')
yolo_model = YOLO('models/yolo/best.pt')
classes = ['Plain', 'Pothole']

@app.route('/')
def index():
    return render_template('index.html', logged_in=('email' in session))

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirmPassword')

    if password != confirm_password:
        return jsonify({'status': 'fail', 'message': 'Passwords do not match.'})

    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    if cursor.fetchone():
        return jsonify({'status': 'fail', 'message': 'Email already exists.'})

    hashed_pw = generate_password_hash(password)
    cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_pw))
    mydb.commit()
    return jsonify({'status': 'success', 'message': 'Registered successfully!'})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    cursor.execute("SELECT password FROM users WHERE email=%s", (email,))
    result = cursor.fetchone()

    if result and check_password_hash(result[0], password):
        session['user'] = email 
        session['email'] = email
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'fail', 'message': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()  # Clears the session
    return jsonify({'status': 'success'})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'email' not in session:
        return jsonify({'status': 'fail', 'message': 'Unauthorized'})

    file = request.files['file']
    if not file or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'status': 'fail', 'message': 'Invalid file format'})

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Image preprocessing
    img_resnet = image.load_img(filepath, target_size=(256, 256))
    img_vgg = image.load_img(filepath, target_size=(224, 224))

    img_resnet_array = image.img_to_array(img_resnet)
    img_resnet_array = np.expand_dims(img_resnet_array, axis=0) / 255.0

    img_vgg_array = image.img_to_array(img_vgg)
    img_vgg_array = np.expand_dims(img_vgg_array, axis=0) / 255.0

    # ResNet50 prediction
    resnet_pred = resnet_model.predict(img_resnet_array)
    resnet_label = classes[np.argmax(resnet_pred)]
    resnet_conf = round(float(np.max(resnet_pred)) * 100, 2)

    # VGG19 prediction
    vgg_pred = vgg_model.predict(img_vgg_array)
    vgg_label = classes[np.argmax(vgg_pred)]
    vgg_conf = round(float(np.max(vgg_pred)) * 100, 2)

    # YOLOv8 detection
    img_cv = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(img_cv, conf=0.5)[0]

    yolo_labels = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{yolo_model.names[cls]} ({conf:.2f})"
        yolo_labels.append(label)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    output_path = os.path.join(PREDICT_FOLDER, filename)
    cv2.imwrite(output_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

    return jsonify({
        'status': 'success',
        'resnet': {'label': resnet_label, 'conf': resnet_conf},
        'vgg': {'label': vgg_label, 'conf': vgg_conf},
        'yolo': yolo_labels if yolo_labels else ['No pothole detected'],
        'image': filename
    })

def run_detection():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 640))
        results = yolo_model.predict(frame_resized, conf=0.25, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{yolo_model.names[cls]} ({conf:.2f})"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Live Detection (ESC to exit)", frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/start-opencv')
def start_opencv():
    if 'email' not in session:
        return jsonify({'status': 'fail', 'message': 'Unauthorized'})

    # Start detection in a background thread
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.start()

    return jsonify({'status': 'success', 'message': 'Detection started'})
@app.route('/check_session', methods=['GET'])
def check_session():
    if 'email' in session:
        return jsonify({'logged_in': True, 'username': session['email']})
    else:
        return jsonify({'logged_in': False}), 401



if __name__ == '__main__':
    app.run(debug=True)