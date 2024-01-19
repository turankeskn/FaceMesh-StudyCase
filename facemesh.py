import numpy as np
import cv2
import mediapipe as mp
from model import train_and_evaluate_skincare_model,generate_synthetic_data

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Function to extract facial landmarks from an image
def extract_landmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run face mesh on the RGB image
    results = face_mesh.process(rgb_image)
    
    # Extract landmarks if available
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
                
    return landmarks

if __name__ == "__main__":
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Synthetic data
    num_samples = 100
    X, y = generate_synthetic_data(num_samples)

    # Train and evaluate the skincare model using k-fold cross-validation
    average_rmse, skincare_model = train_and_evaluate_skincare_model(X, y)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Extract facial landmarks from the frame
        landmarks = extract_landmarks(frame)

        # Flatten and reshape landmarks for prediction
        input_data = np.array([landmarks]).reshape((1, -1))
        
        # Make a prediction using the trained model
        prediction = skincare_model.predict(input_data)

        # Draw face mesh on the frame
        image = frame
        image.flags.writeable = False
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True).process(img)

        image.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec= mp_drawing_styles.get_default_face_mesh_contours_style())

        # Display the frame with landmarks, mesh, and prediction
        cv2.putText(img, f"Skincare Prediction: {prediction[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Skincare Prediction", img)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
