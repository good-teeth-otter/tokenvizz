from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
import json
import os

MODEL_DIR = "/Users/bgadmin/nucleotide-transformer-500m-human-ref"

wp_model = WordPiece.from_file(
    os.path.join(MODEL_DIR, "vocab.txt"),
    unk_token="<unk>",
)
tokenizer = Tokenizer(wp_model)

tokenizer.pre_tokenizer = Whitespace()

tokenizer_json_path = os.path.join(MODEL_DIR, "tokenizer.json")
tokenizer.save(tokenizer_json_path)
print(f"Saved fast tokenizer to {tokenizer_json_path}")

cfg_path = os.path.join(MODEL_DIR, "tokenizer_config.json")
cfg = json.load(open(cfg_path))
cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Updated tokenizer_config.json at {cfg_path}")
