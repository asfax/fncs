import unittest
import os

import nbimporter
import item_based
import user_based


class TestAgent(unittest.TestCase):

    def setUp(self):
        print("Initialised unit test")

    def test_item_based_genre(self):
        userId = 567
        genre = "Comedy"
        ans = item_based.get_rec_item(userId, genre)
        expected = ["Welcome to the Sticks (Bienvenue chez les Ch'tis) (2008)", 'Eddie Izzard: Dress to Kill (1999)', "Dave Chappelle: For What it's Worth (2004)", 'Broadway Danny Rose (1984)',
                    'Everybody Wants Some (2016)', 'Palm Beach Story, The (1942)', 'Zelig (1983)', 'Dish, The (2001)', 'Stir Crazy (1980)', 'Yesterday, Today and Tomorrow (Ieri, oggi, domani) (1963)']
        self.assertEqual(set(expected), set(ans))

    def test_item_based(self):
        userId = 567
        ans = item_based.get_rec_item(userId)
        expected = ['Doctor Who: A Christmas Carol (2010)', 'My Dinner with André (1981)', 'Day of the Doctor, The (2013)', "Guess Who's Coming to Dinner (1967)", 'Five Easy Pieces (1970)', 'Harakiri (Seppuku) (1962)',
                    "Welcome to the Sticks (Bienvenue chez les Ch'tis) (2008)", 'Holy Mountain, The (Montaña sagrada, La) (1973)', 'Eddie Izzard: Dress to Kill (1999)', 'Celebration, The (Festen) (1998)']
        self.assertEqual(set(expected), set(ans))


if __name__ == '__main__':
    main = TestAgent()

    # This executes the unit test/(itself)
    import sys
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgent)
    unittest.TextTestRunner(verbosity=4, stream=sys.stderr).run(suite)
