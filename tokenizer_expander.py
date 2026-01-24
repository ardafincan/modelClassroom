from typing import List

from datasets import load_dataset, Dataset
from sentencepiece import sentencepiece_model_pb2 as sp_model
from tqdm import tqdm
from transformers import AutoTokenizer


class TokenizerExpander:
    def __init__(
        self,
        teacher_tokenizer_id: str,
        student_tokenizer_id: str,
        target_vocab_size: int,
        datasets_dicts: List[dict],
        ref_model_path: str = "/Users/alibayram/Desktop/pythons/distil_trainer_clean/tokenizer_files/custom_tokenizer.model"
    ):
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_id)
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_id)
        self.target_vocab_size = target_vocab_size
        self.datasets_dicts = datasets_dicts
        self.ref_model = sp_model.ModelProto()
        with open(ref_model_path, 'rb') as f:
          self.ref_model.ParseFromString(f.read())

    def __str__(self):
        return f"TokenizerExpander(teacher_tokenizer={self.teacher_tokenizer.name_or_path}, student_tokenizer={self.student_tokenizer.name_or_path}, target_vocab_size={self.target_vocab_size}, datasets_dicts={self.datasets_dicts})"

    def __repr__(self):
        return self.__str__()

    def _load_datasets(self):
        for dataset_dict in tqdm(self.datasets_dicts, desc="Loading datasets"):
            if dataset_dict.get("limit") is not None:
                dataset = load_dataset(dataset_dict["id"], dataset_dict["subset"], split=dataset_dict["split"], streaming=True)
                """ # Get the first example
                print(next(iter(dataset))) """
                rows = []
                for row in tqdm(range(dataset_dict["limit"]), desc="Loading dataset"):
                    rows.append(next(iter(dataset)))
                dataset = Dataset.from_list(rows)
            else:
                dataset = load_dataset(dataset_dict["id"], split=dataset_dict["split"])
            dataset_dict["dataset"] = dataset
        return self.datasets_dicts

    def _select_tokens(self):
        # check if datasets are loaded to datasets_dicts
        if "dataset" not in self.datasets_dicts[0]:
            self._load_datasets()

        freq_dict = {}
        selected_tokens = set()
        for dataset_dict in self.datasets_dicts:
            dataset = dataset_dict["dataset"]
            columns = dataset_dict["columns"]

            for row in tqdm(dataset, desc=f"Processing {dataset_dict.get('id')}"):
                text = " ".join(row[column] if row[column] is not None else "" for column in columns)
                tokens = self.teacher_tokenizer.tokenize(text)
                for token in tokens:
                    freq_dict[token] = freq_dict.get(token, 0) + 1
                    if freq_dict[token] >= dataset_dict["min_freq"]:
                        selected_tokens.add(token)

        return selected_tokens

    def _create_new_tokens_list(self):
        student_vocab = self.student_tokenizer.get_vocab()
        # sort by their ids
        new_tokens = sorted(student_vocab.items(), key=lambda x: x[1])
        new_tokens = [token for token, id in new_tokens]
        new_vocab_size = len(new_tokens)

        teacher_vocab = self.teacher_tokenizer.get_vocab()
        teacher_vocab_size = len(teacher_vocab)

        # first add 1 letter tokens
        for token, id in tqdm(teacher_vocab.items(), desc="Adding 1-letter tokens"):
            if new_vocab_size >= self.target_vocab_size:
                break
            if len(token) == 1 and token not in student_vocab:
                new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                if new_token_id < 300:
                  new_token_id += 300
                new_tokens.insert(new_token_id, token)
                new_vocab_size += 1
                student_vocab[token] = new_vocab_size

        # second add 2 letter tokens
        for token, id in tqdm(teacher_vocab.items(), desc="Adding 2-letter tokens"):
            if new_vocab_size >= self.target_vocab_size:
                break
            if len(token) == 2 and token not in student_vocab:
                new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                if new_token_id < 300:
                  new_token_id += 300
                new_tokens.insert(new_token_id, token)
                new_vocab_size += 1
                student_vocab[token] = new_vocab_size

        # third add selected tokens
        selected_tokens = self._select_tokens()
        for token in tqdm(selected_tokens, desc="Adding selected tokens"):
            if new_vocab_size >= self.target_vocab_size:
                break
            if token not in student_vocab:
                id = teacher_vocab[token]
                new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                if new_token_id < 300:
                  new_token_id += 300
                new_tokens.insert(new_token_id, token)
                new_vocab_size += 1
                student_vocab[token] = new_vocab_size

        # last add remaining 3,4,5 gradually letter tokens until reacheing target vocab size
        for i in range(3, 6):
            if new_vocab_size >= self.target_vocab_size:
                break
            for token, id in tqdm(teacher_vocab.items(), desc=f"Adding {i}-letter tokens"):
                if new_vocab_size >= self.target_vocab_size:
                    break
                if len(token) == i and token not in student_vocab:
                    new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                    if new_token_id < 300:
                        new_token_id += 300
                    new_tokens.insert(new_token_id, token)
                    new_vocab_size += 1

        return new_tokens

    def create_new_tokenizer(self):
        new_tokens = self._create_new_tokens_list()

        model = sp_model.ModelProto()
        model.CopyFrom(self.ref_model)

        del model.pieces[:]

        normal_token_score = -0
        for token in tqdm(new_tokens, desc="Creating sentencepiece model"):
            piece = model.pieces.add()
            piece.piece = token

            added_vocab = self.student_tokenizer.get_added_vocab()

            # Check if the type is CONTROL, UNKNOWN, USER_DEFINED, BYTE
            if token in ["<unk>", "<bos>", "<eos>"]:
                piece.type = 3
                piece.score = 0
            elif token == "<unk>":
                piece.type = 2
                piece.score = 0
            elif token in added_vocab:
                piece.type = 4
                piece.score = 0
            elif token.startswith("<0x"):
                piece.type = 6
                piece.score = 0
            else:
                piece.score = normal_token_score
                piece.type = 1
            # piece.score = normal_token_score
            normal_token_score -= 1

        model.trainer_spec.vocab_size = self.target_vocab_size
        
        with open("new_tokenizer.model", "wb") as f:
            f.write(model.SerializeToString())

        print("New tokenizer created successfully.")
        return "new_tokenizer.model"
