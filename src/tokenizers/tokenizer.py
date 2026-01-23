

class Tokenizer():
    def __init__(self, vocab: dict): #handle unkown, bos, eos tokens 
        """Initiliaze a Tokenizer object
        
        Args:
            self: self
            vocab (dict): vocabulary of Tokenizer with token as key and index as value.
            
        Returns:
            Tokenizer object with properties vocab, merges, size.
        """
        self.vocab = vocab

    @property
    def size(self):
        return len(self.vocab)