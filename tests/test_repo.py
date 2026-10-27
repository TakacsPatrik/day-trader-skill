import json
import os
import unittest

from _load import ROOT


class Repository(unittest.TestCase):
    def test_local_crypto_app_files_exist(self):
        for path in ("app.py", "binance_client.py", "crypto_engine.py", "mtf.py",
                     "ml_model.py", "bot.py", "paper_trading.py", "config.json", "requirements.txt"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, path)), path)

    def test_configuration_has_the_six_default_symbols(self):
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual([coin["symbol"] for coin in config["symbols"]],
                         ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "GRASSUSDT"])
        self.assertEqual(config["intervals"], ["1m", "5m", "15m", "1h"])
        self.assertEqual(config["default_interval"], "1h")
        self.assertEqual(config["ml"]["target_atr"], 4.5)
        self.assertEqual(config["ml"]["stop_atr"], 1.8)
        self.assertEqual(config["ml"]["horizon_bars"], 48)

    def test_legacy_live_broker_and_stock_research_modules_are_removed(self):
        for path in ("trader-broker", "trader-analyst", "trader-pelosi", ".claude-plugin"):
            self.assertFalse(os.path.exists(os.path.join(ROOT, path)), path)

    def test_dashboard_dependencies_are_declared(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
            requirements = handle.read()
        for package in ("streamlit", "plotly", "requests", "scikit-learn"):
            self.assertIn(package, requirements)

    def test_readme_documents_local_only_paper_mode(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
            text = handle.read().lower()
        for phrase in ("streamlit run app.py", "paper trading", "binance", "not financial advice"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
