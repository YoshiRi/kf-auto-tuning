import unittest
import yaml
import os

class TestConfigLoading(unittest.TestCase):
    def setUp(self):
        # Construct the path to the YAML configuration file.
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'kf_auto_tuning', 'sample_config.yaml')

    def test_yaml_loading(self):
        # Ensure the YAML file exists.
        self.assertTrue(os.path.exists(self.config_path), f"Config file not found: {self.config_path}")
        
        # Load the YAML file.
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify that the essential keys exist.
        self.assertIn('system_config', config, "Missing 'system_config' key in YAML file.")
        self.assertIn('optimization_config', config, "Missing 'optimization_config' key in YAML file.")
        
        # Verify some nested configuration keys.
        optimization_config = config['optimization_config']
        self.assertIn('n_calls', optimization_config, "Missing 'n_calls' key in optimization_config.")
        self.assertIn('opt_seed', optimization_config, "Missing 'opt_seed' key in optimization_config.")
if __name__ == '__main__':
    unittest.main()
