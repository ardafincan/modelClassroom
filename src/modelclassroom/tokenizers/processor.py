from .tokenizer import Tokenizer
from ..utils.dictUtils import get_nested
import json

# This class is main process module for Tokenizer class.
class TokenizerProcessor():
    
    def load_tokenizer(self, path: str, vocab_keys: list[str] = ["model", "vocab"], merge_keys: list[str] = ["model", "merges"]):
        f = open(path, "r")
        dict = json.load(f)

        vocab = get_nested(dict, vocab_keys)
        merges = list(get_nested(dict, merge_keys))
        return Tokenizer(vocab, merges)