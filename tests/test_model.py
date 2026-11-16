"""Run:  python -m unittest discover -s tests -v      (or:  pytest)"""
import json
import unittest
import warnings

from startup_model import StartupRiskModel, config


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.m = StartupRiskModel.load()

    def test_output_contract(self):
        r = self.m.predict({"funding_usd_millions": 50, "country": "US", "industry": "fintech"})
        for k in ["success_score", "band", "band_label", "confidence", "dataset_success_rate", "missing_fields",
                  "out_of_range", "funding_percentile", "normalized_input", "notes", "model_version"]:
            self.assertIn(k, r)
        self.assertTrue(0.0 <= r["success_score"] <= 1.0)
        self.assertIn(r["band"], {"failure_like", "mixed", "success_like"})
        json.dumps(r)                                           # must be JSON-serialisable for the API

    def test_ordering_sanity(self):
        big = self.m.predict({"funding_usd_millions": 800, "country": "US", "industry": "fintech"})["success_score"]
        small = self.m.predict({"funding_usd_millions": 0.3, "country": "India", "industry": "edu"})["success_score"]
        self.assertGreater(big, small)

    def test_small_funding_flagged_out_of_range(self):
        r = self.m.predict({"funding_usd_millions": 0.0, "country": "India", "industry": "edu"})
        self.assertTrue(r["out_of_range"]);  self.assertEqual(r["confidence"], "low")

    def test_missing_fields(self):
        r = self.m.predict({"country": "India", "industry": "edu"})
        self.assertEqual(r["missing_fields"], ["funding_usd_millions"]);  self.assertEqual(r["confidence"], "medium")
        r = self.m.predict({"industry": "edu"})
        self.assertEqual(r["confidence"], "low")

    def test_empty_input_does_not_fabricate_a_score(self):
        for bad in [{}, None, "text", {"funding_usd_millions": None, "country": "", "industry": "  "}]:
            r = self.m.predict(bad)
            self.assertIsNone(r["success_score"]);  self.assertEqual(r["band"], "insufficient_data")

    def test_bad_values_do_not_crash(self):
        r = self.m.predict({"funding_usd_millions": "lots", "country": "Narnia", "industry": "underwater basket weaving"})
        self.assertIn("funding_usd_millions", r["missing_fields"]);  self.assertTrue(r["notes"])
        self.assertEqual(r["normalized_input"]["country"], "Other")
        r = self.m.predict({"funding_usd_millions": -5, "country": "US", "industry": "fintech"})
        self.assertIn("funding_usd_millions", r["missing_fields"])
        r = self.m.predict({"funding_usd_millions": 10, "country": "US", "industry": "fintech", "founded_year": 2023, "employees": 4})
        self.assertEqual(r["missing_fields"], [])               # extra keys are ignored

    def test_string_numbers_and_aliases(self):
        a = self.m.predict({"funding_usd_millions": "25", "country": "USA", "industry": "Financial technology"})
        b = self.m.predict({"funding_usd_millions": 25.0, "country": "United States", "industry_bucket": "fintech"})
        self.assertEqual(a["success_score"], b["success_score"])

    def test_batch_equals_single_and_is_deterministic(self):
        rows = [{"funding_usd_millions": 5, "country": "India", "industry": "health"}, {"funding_usd_millions": 300, "country": "China", "industry": "software_ai"}]
        self.assertEqual(self.m.predict_many(rows), [self.m.predict(r) for r in rows])
        self.assertEqual(self.m.predict(rows[0]), self.m.predict(rows[0]))
        self.assertEqual(self.m.predict_many([]), [])

    def test_similar_startups(self):
        s = self.m.similar_startups("quick grocery delivery app with thin margins", industry="commerce_logistics", k_each=3)
        self.assertEqual(set(s), {"failed", "succeeded"})
        self.assertTrue(len(s["failed"]) > 0)
        self.assertTrue(all("name" in x and "similarity" in x for x in s["failed"] + s["succeeded"]))
        json.dumps(s)
        self.assertEqual(self.m.similar_startups("", None), {"failed": [], "succeeded": []})

    def test_retrain_fallback_matches_shipped_model(self):
        rebuilt = StartupRiskModel.retrain(save=False)
        x = {"funding_usd_millions": 40, "country": "UK", "industry": "health"}
        self.assertAlmostEqual(rebuilt.predict(x)["success_score"], self.m.predict(x)["success_score"], places=3)

    def test_card_has_required_metadata(self):
        c = self.m.card
        for k in ["model_version", "sklearn_version", "cv_5fold_x3", "score_bands", "calibration_bins", "limitations"]:
            self.assertIn(k, c)
        self.assertGreater(c["cv_5fold_x3"]["auc"]["mean"], 0.75)       # guard against accidental regressions

if __name__ == "__main__":
    unittest.main()
