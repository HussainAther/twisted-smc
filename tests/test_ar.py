import unittest
from ar.visual import VisualRecognition
from ar.spatial import SpatialRecognition
import cv2

class TestVisualRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the VisualRecognition object once for all test cases.
        """
        cls.visual_recognition = VisualRecognition(
            model_path='yolov3.weights',
            config_path='yolov3.cfg',
            labels_path='coco.names'
        )

    def test_detect_objects(self):
        """
        Test that the object detection model can process an image and detect objects.
        """
        # Create a blank image (black image) for testing
        blank_image = cv2.imread('test_image.jpg')

        detected_objects, _ = self.visual_recognition.detect_objects(blank_image)

        # For this blank image, we expect no objects to be detected
        self.assertIsInstance(detected_objects, list)
        self.assertEqual(len(detected_objects), 0, "No objects should be detected in a blank image.")

    def test_detect_objects_real_image(self):
        """
        Test the object detection with a real image.
        """
        # Use an actual test image (you should provide a sample image with known objects)
        test_image = cv2.imread('test_real_image.jpg')

        detected_objects, _ = self.visual_recognition.detect_objects(test_image)

        # Check if at least one object is detected
        self.assertIsInstance(detected_objects, list)
        self.assertGreater(len(detected_objects), 0, "At least one object should be detected in the real image.")

        # Optionally, check for specific object labels
        labels = [obj['label'] for obj in detected_objects]
        self.assertIn('person', labels, "Expected a person to be detected in the image.")
    

class TestSpatialRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the SpatialRecognition object once for all test cases.
        """
        cls.spatial_recognition = SpatialRecognition()

    def test_get_location(self):
        """
        Test that the spatial recognition can return a simulated GPS location.
        """
        location = self.spatial_recognition.get_location()

        # Check if location is a tuple of (latitude, longitude)
        self.assertIsInstance(location, tuple)
        self.assertEqual(len(location), 2, "Location should be a tuple with two elements (latitude, longitude).")
        self.assertIsInstance(location[0], float)
        self.assertIsInstance(location[1], float)

    def test_reverse_geocode(self):
        """
        Test reverse geocoding to convert GPS coordinates into an address.
        """
        location = self.spatial_recognition.get_location()
        address = self.spatial_recognition.reverse_geocode(location)

        # Check if a valid address is returned
        self.assertIsInstance(address, str)
        self.assertTrue(len(address) > 0, "The reverse geocoded address should not be empty.")

    def test_get_nearby_landmarks(self):
        """
        Test that nearby landmarks are correctly identified within the given radius.
        """
        location = self.spatial_recognition.get_location()
        nearby_landmarks = self.spatial_recognition.get_nearby_landmarks(location, radius_km=1.0)

        # Check if landmarks are returned as a list
        self.assertIsInstance(nearby_landmarks, list)
        self.assertGreater(len(nearby_landmarks), 0, "There should be at least one nearby landmark within the radius.")
    
        # Optionally, check if specific landmarks are found
        self.assertIn("Empire State Building", nearby_landmarks, "Expected 'Empire State Building' to be in the nearby landmarks.")

    def test_calculate_distance(self):
        """
        Test the distance calculation between two locations.
        """
        loc1 = (40.748817, -73.985428)  # Empire State Building
        loc2 = (40.758896, -73.985130)  # Times Square
        
        distance = self.spatial_recognition.calculate_distance(loc1, loc2)

        # Check if the distance is a float and greater than 0
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0, "The distance between two locations should be greater than 0.")
        self.assertLess(distance, 2.0, "The distance between these two locations should be less than 2 km.")


if __name__ == "__main__":
    unittest.main()

