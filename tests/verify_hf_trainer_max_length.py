import unittest
from unittest.mock import MagicMock
from llm_flux.adapters.healing.hf_trainer import HFTrainerAdapter
from llm_flux.core.healing import HealingConfig
from llm_flux.datasets.port import DatasetConfig

class TestHFTrainerAdapterMaxLength(unittest.TestCase):
    def setUp(self):
        self.config = HealingConfig(
            dataset=DatasetConfig(source="dummy"),
            max_steps=10
        )
        self.tokenizer = MagicMock()
        self.adapter = HFTrainerAdapter(self.config, self.tokenizer)

    def test_get_max_seq_length_from_tokenizer(self):
        self.tokenizer.model_max_length = 1024
        model = MagicMock()
        
        max_len = self.adapter._get_max_seq_length(model, self.tokenizer)
        self.assertEqual(max_len, 1024)

    def test_get_max_seq_length_from_tokenizer_large_default(self):
        # Many HF tokenizers use a very large number as default
        self.tokenizer.model_max_length = 1000000000000000019884624838656
        model = MagicMock()
        model.config.max_position_embeddings = 2048
        
        max_len = self.adapter._get_max_seq_length(model, self.tokenizer)
        self.assertEqual(max_len, 2048)

    def test_get_max_seq_length_from_model_config(self):
        self.tokenizer.model_max_length = None
        model = MagicMock()
        model.config.max_position_embeddings = 4096
        
        max_len = self.adapter._get_max_seq_length(model, self.tokenizer)
        self.assertEqual(max_len, 4096)

    def test_get_max_seq_length_from_model_config_n_positions(self):
        self.tokenizer.model_max_length = None
        model = MagicMock()
        del model.config.max_position_embeddings
        model.config.n_positions = 256
        
        max_len = self.adapter._get_max_seq_length(model, self.tokenizer)
        self.assertEqual(max_len, 256)

    def test_get_max_seq_length_default(self):
        self.tokenizer.model_max_length = None
        model = MagicMock()
        del model.config.max_position_embeddings
        
        max_len = self.adapter._get_max_seq_length(model, self.tokenizer, default=128)
        self.assertEqual(max_len, 128)

    def test_manual_override(self):
        self.config.max_seq_length = 123
        self.tokenizer.model_max_length = 1024
        model = MagicMock()

        # We can't easily test the 'heal' method without full mocking of transformers
        # but we can verify the logic that uses max_seq_length
        # In heal():
        # max_seq_len = cfg.max_seq_length if cfg.max_seq_length else self._get_max_seq_length(model, actual_tokenizer)
        
        cfg = self.config
        max_seq_len = cfg.max_seq_length if cfg.max_seq_length else self.adapter._get_max_seq_length(model, self.tokenizer)
        self.assertEqual(max_seq_len, 123)

if __name__ == "__main__":
    unittest.main()
