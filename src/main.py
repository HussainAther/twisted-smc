from nlp.model import LanguageModel
from nlp.twisted_smc import TwistedSMC
from ar.visual import VisualRecognition
from ar.spatial import SpatialRecognition

import cv2

def main():
    # Initialize the NLP model and Twisted SMC engine
    language_model = LanguageModel()
    twisted_smc = TwistedSMC(num_particles=100, proposal_dist=None, twist_function=None)  # Placeholder for SMC functions
    
    # Initialize the Visual and Spatial recognition modules
    visual_recognition = VisualRecognition(
        model_path='yolov3.weights', 
        config_path='yolov3.cfg', 
        labels_path='coco.names'
    )
    spatial_recognition = SpatialRecognition()

    # Start the video capture for AR
    cap = cv2.VideoCapture(0)

    print("AR Navigation System is running...")
    print("Press 'q' to quit.")

    while True:
        # Capture the current frame from the video stream (simulating AR view)
        ret, frame = cap.read()
        if not ret:
            print("Error capturing video frame.")
            break

        # Detect objects in the current frame using the visual recognition module
        detected_objects, frame_with_boxes = visual_recognition.detect_objects(frame)

        # Display the current frame with bounding boxes drawn around detected objects
        cv2.imshow("AR View - Object Detection", frame_with_boxes)

        # Get the current spatial (GPS) location of the user
        current_location = spatial_recognition.get_location()
        nearby_landmarks = spatial_recognition.get_nearby_landmarks(current_location, radius_km=1.0)
        print(f"Nearby landmarks: {nearby_landmarks}")

        # User input (simulate text query input)
        query = input("Ask something (e.g., 'What's special here?'): ")

        # Generate multiple probabilistic interpretations using twisted SMC (placeholder)
        print("Interpreting query with multiple possible outcomes...")
        probabilistic_interpretations = language_model.get_probabilistic_interpretations(query, num_samples=3)
        print(f"Possible interpretations: {probabilistic_interpretations}")

        # Example: Use visual context (landmarks) to refine query interpretation
        if nearby_landmarks:
            refined_interpretation = language_model.interpret_with_context(query, {"landmark": nearby_landmarks[0]})
            print(f"Refined interpretation with context: {refined_interpretation}")
        
        # Press 'q' to quit the loop and end the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

