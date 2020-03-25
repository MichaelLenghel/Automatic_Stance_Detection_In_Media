import unittest
import sys

class test_generate_lda_model(unittest.TestCase):
    def setUp(self):
        sys.path.append('..')
        import generate_lda_model as glm
        self.glm = glm

    def test_multiple_replacements(self):
        # Check replacement method works correctly
        self.assertEqual(self.glm.multiple_replacements("Dann is happy - without = symbols"), "Dann is happy  without  symbols")
    
    def test_remove_stopwords(self):
        self.assertEqual(self.glm.remove_stopwords([","]), [[]])
    
    def test_lemmatize_words(self):
        self.assertEqual(self.glm.lemmatization([["Doggy"], ["Run"], ["Walk"], ["Day"]]), [[], ['run'], ['walk'], ['day']])


if __name__ == '__main__':
    unittest.main()