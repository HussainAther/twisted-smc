import cv2
import numpy as np

class VisualRecognition:
    def __init__(self, model_path='yolov3.weights', config_path='yolov3.cfg', labels_path='coco.names'):
        """
        Initialize the object detection model (YOLO in this case) using OpenCV.
        
        Args:
            model_path (str): Path to the pre-trained weights for the detection model.
            config_path (str): Path to the model configuration file.
            labels_path (str): Path to the file containing labels for the detected objects.
        """
        # Load YOLO model
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load the labels (e.g., COCO dataset labels)
        with open(labels_path, 'r') as f:
            self.labels = f.read().strip().split("\n")
        
        # Set up color scheme for detected object bounding boxes
        self.colors = np.random.uniform(0, 255, size=(len(self.labels), 3))

    def detect_objects(self, frame):
        """
        Perform object detection on the given video frame.

        Args:
            frame (numpy.ndarray): The image frame from the video feed (or camera).
        
        Returns:
            detected_objects (list): A list of detected objects, each containing:
                - label (str): The object label (e.g., "person", "car").
                - confidence (float): The confidence score for the detection.
                - bounding_box (tuple): The bounding box coordinates (x, y, w, h).
        """
        height, width = frame.shape[:2]
        
        # Preprocess the frame for YOLO (resize and normalization)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        
        # Perform forward pass through YOLO network
        detections = self.net.forward(self.output_layers)
        
        # Initialize lists for detected objects and bounding boxes
        detected_objects = []
        boxes = []
        confidences = []
        class_ids = []
        
        # Process detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Filter weak detections
                    # Object detected
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maxima suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Collect the final detected objects after NMS
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = self.labels[class_ids[i]]
            confidence = confidences[i]
            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "bounding_box": (x, y, w, h)
            })
            
            # Draw the bounding box and label on the image (for visualization)
            color = self.colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detected_objects, frame

    def recognize_landmark(self, detected_objects):
        """
        A placeholder function to map detected objects to specific landmarks (e.g., known buildings or monuments).
        This can be extended with a database of known landmarks.

        Args:
            detected_objects (list): The list of detected objects from the video frame.

        Returns:
            landmark (str): The identified landmark, if any.
        """
        # Example: Check if certain objects (like 'building', 'sign', etc.) correspond to landmarks
        landmark = None
        for obj in detected_objects:
            if obj['label'] in ['building', 'statue', 'monument']:  # Example labels
                landmark = obj['label']
                break
        
        return landmark

# Example usage
if __name__ == "__main__":
    # Initialize the visual recognition module
    visual_recognition = VisualRecognition(
        model_path='yolov3.weights',
        config_path='yolov3.cfg',
        labels_path='coco.names'
    )

    # Open a video stream (using your device's camera)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the current frame
        detected_objects, frame_with_boxes = visual_recognition.detect_objects(frame)

        # Display the video feed with bounding boxes
        cv2.imshow('Object Detection', frame_with_boxes)

        # Example: Check if any landmarks are recognized
        landmark = visual_recognition.recognize_landmark(detected_objects)
        if landmark:
            print(f"Landmark detected: {landmark}")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

