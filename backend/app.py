from flask import Flask, Response, render_template
from flask_cors import CORS
from tflite import movenet, draw_prediction_on_image
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

def generate_frames():
    cap = cv2.VideoCapture(0)
    input_size = 256

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
       
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (input_size, input_size))
        input_image = np.expand_dims(input_image, axis=0)
        
        keypoints_with_scores = movenet(input_image)

        frame_with_keypoints = draw_prediction_on_image(frame, keypoints_with_scores)

        ret, buffer = cv2.imencode('.jpg', frame_with_keypoints)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)