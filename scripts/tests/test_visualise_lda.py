import unittest
import sys

class test_visualise_lda(unittest.TestCase):
    def setUp(self):
        sys.path.append('..')
        import visualise_lda as vl
        self.vl = vl
        self.median_pol = [1, 3, 3, 4, 5]
        self.median_obj = [10, 11, 12, 12, 14]


    def test_getSentiment_polarity(self):
        # Test if sujective in range from - 1 to 1 where -1 is negative and 1 is positive

        # Negative test case
        self.assertLess(self.vl.getSentiment('This referendum is terrible')[0], 0)
        # Positive test case
        self.assertGreater(self.vl.getSentiment('This referendum is amazing')[0], 0.5)
    
    def test_getSentiment_objectivity(self):
        # Test if objecvtive in range of 0 to 1 where 0 is objective and 1 is opinion

        # Negative test case
        self.assertEqual(self.vl.getSentiment('This referendum is terrible')[1], 1)
        # Positive test case
        self.assertLess(self.vl.getSentiment('This referendum exists')[1], 0.5)
    
    def test_retrieve_word_corpus(self):
        # Ensure does not return an empty list
        self.assertNotEqual(self.vl.retrieve_word_corpus, [])
    
    def test_calculate_sentiment_median(self):
        # Test polarity median
        self.assertEqual(self.vl.calculate_sentiment_median(self.median_pol, self.median_obj)[0], 3)

        # Test subjectivity median
        self.assertEqual(self.vl.calculate_sentiment_median(self.median_pol, self.median_obj)[1], 12)
    
    def test_calculate_sentiment_mode(self):
        avg_pol = sum(self.median_pol) / len(self.median_pol) 
        avg_obj = sum(self.median_obj) / len(self.median_obj)

        # Test polarity median
        self.assertEqual(self.vl.calculate_sentiment_mode(self.median_pol, self.median_obj, avg_pol, avg_obj)[0], 3)

        # Test subjectivity median
        self.assertEqual(self.vl.calculate_sentiment_mode(self.median_pol, self.median_obj, avg_pol, avg_obj)[1], 12)
    
    def test_get_topic_sentences(self):
        # Check filters topic
        self.assertEqual(self.vl.get_topic_sentences('Brexit', ['Brexit:::Auther:::Date:::Brexit is something', 'China:::Auther:::Date:::China is something']), ['Brexit is something'])



       


if __name__ == '__main__':
    unittest.main()