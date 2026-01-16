from .tokenizer import Tokenizer
from ..utils.dictUtils import get_nested
import json

# This class is main process module for Tokenizer class.
class TokenizerProcessor():

    def load_tokenizer(self, path: str, vocab_keys: list[str] = ["model", "vocab"], merge_keys: list[str] = ["model", "merges"]):
        """ This function parses tokenizer.json from path and store it as a Tokenizer object.
        
        Args: 
            self: self
            path (str): Path for json file to parse
            vocab_keys (list[str]): Keys to vocabulary in a nested dictionary. Defaults to ["model", "vocab"]
            merge_keys (list[str]): Keys to merges in a nested dictionary. Defaults to ["model", "merges"]
        Returns:
            A Tokenizer object with vocab and merges. 
        """
        f = open(path, "r")
        tokenizerDict = json.load(f)
        f.close()
        vocab = get_nested(tokenizerDict, vocab_keys)
        merges = list(get_nested(tokenizerDict, merge_keys))
        return Tokenizer(vocab, merges)
    
        