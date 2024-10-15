import unittest
from nlp.model import LanguageModel

class TestLanguageModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialize the language model once for all test cases.
        This is done once per test class to avoid re-loading the model for each test.
        """
        cls.language_model = LanguageModel(model_name='facebook/bart-large')

    def test_generate_response(self):
        """
        Test that the language model can generate a valid response to a basic query.
        """
        query = "What is the capital of France?"
        response = self.language_model.generate_response(query)

        # Check if response is a non-empty string
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0, "The generated response should not be empty.")

        # Optionally, check if the response contains expected content
        self.assertIn("Paris", response, "Expected 'Paris' to be in the response for the query.")

    def test_get_probabilistic_interpretations(self):
        """
        Test that the language model can generate multiple interpretations for a query.
        """
        query = "What is special here?"
        interpretations = self.language_model.get_probabilistic_interpretations(query, num_samples=3)

        # Check if the result is a list of strings with the correct length
        self.assertIsInstance(interpretations, list)
        self.assertEqual(len(interpretations), 3, "Should generate exactly 3 probabilistic interpretations.")
        
        for interpretation in interpretations:
            self.assertIsInstance(interpretation, str)
            self.assertTrue(len(interpretation) > 0, "Each interpretation should be a non-empty string.")

    def test_interpret_with_context(self):
        """
        Test that the language model correctly integrates context into query interpretation.
        """
        query = "What is special here?"
        context = {"landmark": "Empire State Building"}

        refined_interpretation = self.language_model.interpret_with_context(query, context)

        # Check if the response is a non-empty string
        self.assertIsInstance(refined_interpretation, str)
        self.assertTrue(len(refined_interpretation) > 0, "The refined interpretation should not be empty.")

        # Check if the response incorporates the context (e.g., mentions the landmark)
        self.assertIn("Empire State Building", refined_interpretation, "The response should mention the landmark from the context.")

if __name__ == "__main__":
    unittest.main()

