import unittest
import os

import nbimporter
import item_based
import user_based


class TestAgent(unittest.TestCase):

    def setUp(self):
        print("Initialised unit test")

    def test_user_based_genre(self):
        userId = 67
        genre = "Drama"
        ans = user_based.get_rec_user(userId, genre)
        expected = ['Kids (1995)', 'Five Senses, The (1999)', 'Little Women (1994)', 'Sweet Hereafter, The (1997)', 'Inherit the Wind (1960)', "One Flew Over the Cuckoo's Nest (1975)",
                    'Thirteen Conversations About One Thing (a.k.a. 13 Conversations) (2001)', 'Requiem for a Dream (2000)', '12 Angry Men (1957)', 'River Runs Through It, A (1992)']
        self.assertEqual(set(expected), set(ans))

    def test_user_based(self):
        userId = 67
        ans = user_based.get_rec_user(userId)
        expected = ["One Flew Over the Cuckoo's Nest (1975)", 'Godfather, The (1972)', 'Goodfellas (1990)', 'Sweet Hereafter, The (1997)', 'Never Cry Wolf (1983)',
                    'Godfather: Part II, The (1974)', 'Born Yesterday (1950)', 'Three Billboards Outside Ebbing, Missouri (2017)', '12 Angry Men (1957)', 'Five Senses, The (1999)']
        self.assertEqual(set(expected), set(ans))


if __name__ == '__main__':
    main = TestAgent()

    # This executes the unit test/(itself)
    import sys
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgent)
    unittest.TextTestRunner(verbosity=4, stream=sys.stderr).run(suite)
