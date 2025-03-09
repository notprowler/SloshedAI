from tflite import movenet, draw_prediction_on_image
import  cv2
import numpy as np

cap = cv2.VideoCapture(0)  
input_size = 256 

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    # Resizing img to meet the model's input size
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (input_size, input_size))
    input_image = np.expand_dims(input_image, axis=0)

    keypoints_with_scores = movenet(input_image)

    frame_with_keypoints = draw_prediction_on_image(frame, keypoints_with_scores)

    cv2.namedWindow("MoveNet Pose Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MoveNet Pose Detection", 1280, 720) 
    cv2.imshow("MoveNet Pose Detection", frame_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()