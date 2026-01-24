from typing import List, Dict

from datasets import load_dataset, Dataset
from sentencepiece import sentencepiece_model_pb2 as sp_model
from tqdm import tqdm
from transformers import AutoTokenizer


# Special tokens with their fixed IDs (from added_tokens.json)
SPECIAL_TOKENS: Dict[str, int] = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "[multimodal]": 5,
}

# Image soft token is the last token with a specific ID
IMAGE_SOFT_TOKEN = "<image_soft_token>"
IMAGE_SOFT_TOKEN_ID = 131072


class TokenizerExpander:
    def __init__(
        self,
        teacher_tokenizer_id: str,
        student_tokenizer_id: str,
        target_vocab_size: int,
        datasets_dicts: List[dict],
        ref_model_path: str = "/Users/alibayram/Desktop/pythons/distil_trainer_clean/tokenizer_files/tokenizer.model"
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

    def _get_piece_type(self, token: str) -> int:
        """
        Determine the SentencePiece type for a token.
        Types:
            1 = NORMAL
            2 = UNKNOWN
            3 = CONTROL
            4 = USER_DEFINED
            6 = BYTE
        """
        if token == "<unk>":
            return 2  # UNKNOWN type
        if token in SPECIAL_TOKENS or token == IMAGE_SOFT_TOKEN:
            return 3  # CONTROL type
        if token.startswith("<unused"):
            return 3  # CONTROL type for unused tokens
        if token.startswith("<0x"):
            return 6  # BYTE type
        
        added_vocab = self.student_tokenizer.get_added_vocab()
        if token in added_vocab:
            return 4  # USER_DEFINED type
        
        return 1  # NORMAL type

    def _build_final_token_list(self, base_tokens: List[str]) -> List[str]:
        """
        Build the final token list ensuring special tokens are at their fixed positions.
        
        The structure is:
        - IDs 0-5: Special tokens (<pad>, <eos>, <bos>, <unk>, <mask>, [multimodal])
        - IDs 6 onwards: Regular tokens and unused tokens
        - ID 131072: <image_soft_token>
        """
        # Remove special tokens from base_tokens if they exist
        special_token_set = set(SPECIAL_TOKENS.keys())
        special_token_set.add(IMAGE_SOFT_TOKEN)
        
        filtered_tokens = [t for t in base_tokens if t not in special_token_set and not t.startswith("<unused")]
        
        # Build final list with correct positions
        final_tokens = [""] * (IMAGE_SOFT_TOKEN_ID + 1)
        
        # Place special tokens at their fixed positions
        for token, token_id in SPECIAL_TOKENS.items():
            final_tokens[token_id] = token
        
        # Place <image_soft_token> at the last position
        final_tokens[IMAGE_SOFT_TOKEN_ID] = IMAGE_SOFT_TOKEN
        
        # Calculate how many unused tokens we need
        # Between special tokens (0-5) and image_soft_token (131072)
        num_regular_slots = IMAGE_SOFT_TOKEN_ID - len(SPECIAL_TOKENS)  # 131072 - 6 = 131066
        num_unused_needed = num_regular_slots - len(filtered_tokens)
        
        # Fill positions 6 onwards with filtered tokens
        current_pos = len(SPECIAL_TOKENS)  # Start at position 6
        for token in filtered_tokens:
            if current_pos >= IMAGE_SOFT_TOKEN_ID:
                break
            final_tokens[current_pos] = token
            current_pos += 1
        
        # Fill remaining positions with unused tokens
        unused_counter = 0
        while current_pos < IMAGE_SOFT_TOKEN_ID:
            unused_token = f"<unused{unused_counter}>"
            final_tokens[current_pos] = unused_token
            current_pos += 1
            unused_counter += 1
        
        return final_tokens

    def create_new_tokenizer(self):
        base_tokens = self._create_new_tokens_list()
        
        # Build final token list with correct positions
        new_tokens = self._build_final_token_list(base_tokens)

        model = sp_model.ModelProto()
        model.CopyFrom(self.ref_model)

        del model.pieces[:]

        normal_token_score = 0
        for idx, token in enumerate(tqdm(new_tokens, desc="Creating sentencepiece model")):
            piece = model.pieces.add()
            piece.piece = token
            piece.type = self._get_piece_type(token)
            
            # Special tokens and control tokens have score 0
            if piece.type in [2, 3, 4, 6]:  # UNKNOWN, CONTROL, USER_DEFINED, BYTE
                piece.score = 0
            else:
                piece.score = normal_token_score
                normal_token_score -= 1

        model.trainer_spec.vocab_size = len(new_tokens)
        
        with open("new_tokenizer.model", "wb") as f:
            f.write(model.SerializeToString())

        # Print summary
        print(f"\nNew tokenizer created successfully!")
        print(f"Total vocab size: {len(new_tokens)}")
        print(f"\nSpecial tokens verification:")
        for token, expected_id in SPECIAL_TOKENS.items():
            actual_id = new_tokens.index(token) if token in new_tokens else -1
            status = "✓" if actual_id == expected_id else "✗"
            print(f"  {status} {token}: expected={expected_id}, actual={actual_id}")
        
        # Verify image_soft_token
        img_actual_id = new_tokens.index(IMAGE_SOFT_TOKEN) if IMAGE_SOFT_TOKEN in new_tokens else -1
        img_status = "✓" if img_actual_id == IMAGE_SOFT_TOKEN_ID else "✗"
        print(f"  {img_status} {IMAGE_SOFT_TOKEN}: expected={IMAGE_SOFT_TOKEN_ID}, actual={img_actual_id}")
        
        return "new_tokenizer.model"
