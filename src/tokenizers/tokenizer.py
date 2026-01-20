from .merge import Merge

class Tokenizer():
    def __init__(self, vocab: dict, merges: dict[str, Merge]): #redesign merges structure
        """Initiliaze a Tokenizer object
        
        Args:
            self: self
            vocab (dict): vocabulary of Tokenizer with token as key and index as value.
            merges (list[str]): merges list of Tokenizer as list of strings.
            
        Returns:
            Tokenizer object with properties vocab, merges, size.
        """
        self.vocab = vocab
        self.merges = merges

    @property
    def size(self):
        return len(self.vocab)