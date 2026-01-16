from .tokenizer import Tokenizer
from ..utils.dictUtils import get_nested
import json

# This class is main process module for Tokenizer class.
class TokenizerProcessor():

    def load_tokenizer(self, path: str, vocab_keys: list[str] = ["model", "vocab"], merge_keys: list[str] = ["model", "merges"]):
        """This function parses tokenizer.json from path and store it as a Tokenizer object.
        
        Args: 
            self: self
            path (str): Path for json file to parse
            vocab_keys (list[str]): Keys to vocabulary in a nested dictionary. Defaults to ["model", "vocab"]
            merge_keys (list[str]): Keys to merges in a nested dictionary. Defaults to ["model", "merges"]

        Returns:
            A Tokenizer object with vocab and merges. 
        """
        with open(path, "r") as f:
            tokenizerDict = json.load(f)
        vocab = get_nested(tokenizerDict, vocab_keys)
        merges = list(get_nested(tokenizerDict, merge_keys))
        return Tokenizer(vocab, merges)

    def store_tokenizer(self, path: str, tokenizer: Tokenizer, vocab_keys: list[str] = ["model", "vocab"],
                        merge_keys: list[str] = ["model", "merges"], target_path: str = ""): 
        """This function stores the given Tokenizer object as tokenizer.json file. 
        ATTENTION: path parameter should point the json file of original tokenizer that user want to change.
        
        Args:
            self: self
            path (str): Path for json file to write.
            tokenizer (Tokenizer): Tokenizer to save.
            
        Returns: 
            The path of the saved Tokenizer, if successfull."""
        
        with open(path, "r") as f:
            tokenizerDict = json.load(f)

        vocab_parent = get_nested(tokenizerDict, vocab_keys[:-1])
        vocab_parent[vocab_keys[-1]] = tokenizer.vocab
        merges_parent = get_nested(tokenizerDict, merge_keys[:-1])
        merges_parent[merge_keys[-1]] = tokenizer.merges
        
        path = path if (target_path == "") else target_path
        with open(path, "w") as f:
            json.dump(tokenizerDict, f)
        
        return path
    
    def unify_tokenizers(self, source: Tokenizer, target: Tokenizer, target_size: int, careTurkish: bool = False):
        """This function unifies two distinct tokenizers with different sizes and vocabulary.
        
        Args:
            self: self
            source (Tokenizer): Tokenizer of teacher model, source tokenizer. Tokenizer that will give tokens to target.
            target (Tokenizer): Tokenizer of student model, target tokenizer. Tokenizer that will take new tokens from source.
            target_size (int): Size of target tokenizer.
            
        Returns:
            A unified Tokenizer object that took new tokens from source.
        """    
        source_tokens = source.vocab.keys()
        target_tokens = target.vocab.keys()
        
        #Transferring each one character token first
        for idx, token in enumerate(source_tokens):
            if token not in target_tokens:
                if len(token) == 1:
                    target.vocab.update({token, source.vocab[token]})
                elif len(token) == 2:
                    target.vocab.update({token, source.vocab[token]})
                    #Add merges for token but persist order. 
                    target.merges.append(source.merges[idx]) # this line probably will change
                else:
                    #I will implement here next
                    continue
            



