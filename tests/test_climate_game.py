import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import unittest
import numpy as np
from climate_game import GameParams, solve_model, PLAYERS

class TestClimateGameHeterogeneity(unittest.TestCase):
    def setUp(self):
        """Initialize default parameters for testing."""
        self.params = GameParams(T=2)  # Short horizon for speed

    def test_lambda_impact(self):
        """
        Verify that changing a specific player's lambda alters their 
        adoption probability while others remain consistent (ceteris paribus).
        """
        # Scenario 1: High rationality for US
        self.params.lam["US"] = 100.0  # US is near-determinstic
        V1, sigma1, _, _ = solve_model(self.params)
        
        # Scenario 2: High noise for US
        self.params.lam["US"] = 0.01   # US is basically flipping coins
        V2, sigma2, _, _ = solve_model(self.params)
        
        # Check initial state (0,0,0,0) at t=1
        start_state = (0, 0, 0, 0)
        prob_rational = sigma1[1][start_state][0] # US index is 0
        prob_noisy = sigma2[1][start_state][0]
        
        # Assertion: The probabilities should differ significantly
        self.assertNotAlmostEqual(prob_rational, prob_noisy, places=4,
            msg="Changing US lambda did not affect US adoption probability.")

    def test_asymmetric_rationality_solver(self):
        """
        Ensure the solver doesn't crash with highly asymmetric lambdas
        and that it correctly maps the lambda dict to the correct player.
        """
        self.params.lam = {
            "US": 10.0, 
            "EU": 0.1, 
            "CN": 5.0, 
            "RoW": 1.0
        }
        
        try:
            V, sigma, _, _ = solve_model(self.params)
        except KeyError as e:
            self.fail(f"solve_model raised KeyError: {e}. Check if player names match dict keys.")
        except Exception as e:
            self.fail(f"solve_model crashed with asymmetric lambdas: {e}")

        # Basic integrity check on the returned sigma table
        state = (0, 0, 0, 0)
        self.assertTrue(0 <= sigma[1][state][0] <= 1, "US probability out of bounds")
        self.assertTrue(0 <= sigma[1][state][1] <= 1, "EU probability out of bounds")

    def test_parameter_isolation(self):
        """
        Ensure that the lambda of one player is actually used for that player.
        """
        test_state = (0, 0, 1, 1)
        
        # Set EVERYONE to be noisy first
        for p in PLAYERS:
            self.params.lam[p] = 0.0001
            
        # Now make ONLY the US hyper-rational
        self.params.lam["US"] = 100.0
        
        V, sigma, QA, QD = solve_model(self.params)
        
        # EU is still noisy, and since everyone else is noisy, 
        # there's no strong strategic pull. It should be ~0.5.
        eu_prob = sigma[1][test_state][1]
        self.assertAlmostEqual(eu_prob, 0.5, places=2, 
            msg="EU with tiny lambda should have adoption probability near 0.5")
        
        # US is rational and should be near-deterministic (0 or 1)
        us_prob = sigma[1][test_state][0]
        self.assertTrue(us_prob > 0.99 or us_prob < 0.01, 
            f"US with high lambda should be deterministic, got {us_prob}")

    if __name__ == "__main__":
        unittest.main()