class Tokenizer():
    def __init__(self, vocab: dict , merges: list[str]):
        self.vocab = vocab
        self.merges = merges

        @property
        def size(self):
            return len(vocab)