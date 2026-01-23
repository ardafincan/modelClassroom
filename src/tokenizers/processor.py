from .tokenizer import Tokenizer
from ..utils.utils import get_nested
import json
from sentencepiece import sentencepiece_model_pb2 as sp_model
from ..utils.freq_utils.freq import func


# This class is main process module for Tokenizer class.
class TokenizerProcessor:
    def load_tokenizer(self, path: str, vocab_keys: list[str] = ["model", "vocab"]):
        """This function parses tokenizer.json from path and store it as a Tokenizer object.

        Args:
            self: self
            path (str): Path for json file to parse
            vocab_keys (list[str]): Keys to vocabulary in a nested dictionary. Defaults to ["model", "vocab"]

        Returns:
            A Tokenizer object with vocab.
        """
        with open(path, "r") as f:
            tokenizerDict = json.load(f)

        vocab = get_nested(tokenizerDict, vocab_keys)
        return Tokenizer(vocab)

    def store_tokenizer(
        self,
        path: str,
        tokenizer: Tokenizer,
        vocab_keys: list[str] = ["model", "vocab"],
        tokenizer_config: dict = {},
    ):
        """This function stores the given Tokenizer object as tokenizer.json file.
        ATTENTION: path parameter should point the json file of original tokenizer that user want to change.

        Args:
            self: self
            path (str): Path for json file to write.
            tokenizer (Tokenizer): Tokenizer to save.

        Returns:
            The path of the saved Tokenizer, if successfull."""

        model = sp_model.ModelProto()

        if tokenizer_config == {}:
            model.trainer_spec.model_type = sp_model.TrainerSpec.ModelType.BPE
            model.trainer_spec.vocab_size = tokenizer.size

            model.normalizer_spec.name = "identity"
            model.normalizer_spec.add_dummy_prefix = True
            model.normalizer_spec.remove_extra_whitespaces = True
        # implement else here

        for idx, token in enumerate(tokenizer.vocab):
            piece = model.SentencePiece()
            piece.piece = token
            piece.score = -idx
            piece.type = model.SentencePiece.NORMAL
            model.pieces.append(piece)

        with open(path, "wb") as f:
            f.write(model.SerializeToString())

        return path

    def unify_tokenizers(
        self,
        source: Tokenizer,
        target: Tokenizer,
        target_size: int,
        langSpecificList: list = [],
    ):
        """This function unifies two distinct tokenizers with different sizes and vocabulary.

        Args:
            self: self
            source (Tokenizer): Tokenizer of teacher model, source tokenizer. Tokenizer that will give tokens to target.
            target (Tokenizer): Tokenizer of student model, target tokenizer. Tokenizer that will take new tokens from source.
            target_size (int): Size of target tokenizer.

        Returns:
            A unified Tokenizer object that took new tokens from source.
        """
        sourceList = list(source.vocab.keys())
        targetList = list(target.vocab.keys())

        targetListSize = len(targetList)

        langSpecificList = func(source)

        for token in sourceList:
            if targetListSize >= target_size:
                break
            if token not in targetList and len(token) == 1:
                targetList.insert(
                    source.vocab[token], token
                )  # fix indexing for target tokenizer
                targetListSize += 1

        for token in sourceList:
            if targetListSize >= target_size:
                break
            if token not in targetList and len(token) == 2:
                targetList.insert(source.vocab[token], token)
                targetListSize += 1

        if not langSpecificList == []:
            for token in langSpecificList:
                if targetListSize >= target_size:
                    break
                if token not in targetList:
                    targetList.insert(source.vocab[token], token)
                    targetListSize += 1

        i = 3
        while targetListSize < target_size:
            if (
                targetListSize >= target_size or source.size + target.size < target_size
            ):  # fix here
                break
            for token in sourceList:
                if targetListSize >= target_size:
                    break
                if token not in targetList and len(token) == i:
                    targetList.insert(source.vocab[token], token)
                    targetListSize += 1
            i += 1

        target_vocab = {token: idx for idx, token in enumerate(targetList)}
        return Tokenizer(target_vocab)
