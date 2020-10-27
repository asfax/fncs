import unittest

import nbimporter
import item_based
import user_based
import ensemble


class TestAgent(unittest.TestCase):

    def setUp(self):
        print("Initialised unit test")

    def test_ensemble_genre(self):
        userId = 27
        genre = "Action"
        ans = ensemble.ensemble(userId, genre)
        expected = ['13 Assassins (Jûsan-nin no shikaku) (2010)',
                    'Double Team (1997)', 'Crows Zero (Kurôzu zero) (2007)',
                    'Rapid Fire (1992)', 'Game of Death (1978)',
                    'Double Impact (1991)', 'Kiss of the Dragon (2001)',
                    'Raw Deal (1986)', 'Thunderbolt and Lightfoot (1974)',
                    'Minnie and Moskowitz (1971)']
        self.assertEqual(set(expected), set(ans))

    def test_ensemble(self):
        userId = 27
        ans = ensemble.ensemble(userId)
        expected = ['Lamerica (1994)',
                    'Autumn Sonata (Höstsonaten) (1978)',
                    'Entertaining Angels: The Dorothy Day Story (1996)',
                    'Streetcar Named Desire, A (1951)',
                    'Story of Women (Affaire de femmes, Une) (1988)',
                    'Best of Youth, The (La meglio gioventù) (2003)',
                    'Son of Rambow (2007)', 'Beautiful People (1999)',
                    'Secrets & Lies (1996)', 'Bliss (2012)',
                    'My Dinner with André (1981)',
                    'Anne of Green Gables (1985)',
                    'Passenger,The (Professione: reporter) (1975)',
                    'Elling (2001)', 'Last Picture Show, The (1971)',
                    "Don't Move (Non ti muovere) (2004)",
                    'Trial, The (Procès, Le) (1962)',
                    "Swept Away (Travolti da un insolito destino "
                    "nell'azzurro mare d'Agosto) (1975)",
                    'Marriage of Maria Braun, The '
                    '(Ehe der Maria Braun, Die) (1979)']
        self.assertEqual(set(expected), set(ans))


if __name__ == '__main__':
    main = TestAgent()

    # This executes the unit test/(itself)
    import sys
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgent)
    unittest.TextTestRunner(verbosity=4, stream=sys.stderr).run(suite)
