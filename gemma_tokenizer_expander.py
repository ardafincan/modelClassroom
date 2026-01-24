from typing import List, Optional

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_model
from tqdm import tqdm
from transformers import GemmaTokenizerFast

# Default reference model from HuggingFace Hub
DEFAULT_REF_MODEL_ID = "google/gemma-3-1b-it"

FIRST_ADDED_TOKENS = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<mask>",
    "[multimodal]",
    *[f"<unused{i}>" for i in range(100)],
    "<start_of_turn>",
    "<end_of_turn>",
    "<table>",
    "<caption>",
    "<thead>",
    "<tbody>",
    "<tfoot>",
    "<tr>",
    "<th>",
    "<td>",
    "</table>",
    "</caption>",
    "</thead>",
    "</tbody>",
    "</tfoot>",
    "</tr>",
    "</th>",
    "</td>",
    "<h1>",
    "<h2>",
    "<h3>",
    "<h4>",
    "<h5>",
    "<h6>",
    "<blockquote>",
    "</h1>",
    "</h2>",
    "</h3>",
    "</h4>",
    "</h5>",
    "</h6>",
    "</blockquote>",
    "<strong>",
    "<em>",
    "<b>",
    "<i>",
    "<u>",
    "<s>",
    "<sub>",
    "<sup>",
    "<code>",
    "</strong>",
    "</em>",
    "</b>",
    "</i>",
    "</u>",
    "</s>",
    "</sub>",
    "</sup>",
    "</code>",
    "<a>",
    "<html>",
    "<body>",
    "<img>",
    "<span>",
    "<bbox>",
    "<ul>",
    "<li>",
    "<div>",
    "<iframe>",
    "<footer>",
    "</a>",
    "</html>",
    "</body>",
    "</img>",
    "</span>",
    "</bbox>",
    "</ul>",
    "</li>",
    "</div>",
    "</iframe>",
    "</footer>",
    *[f"<0x{i:02X}>" for i in range(256)],
]

LAST_ADDED_TOKENS = [
    "<start_of_image>",
    "<end_of_image>",
    *[f"<unused{i}>" for i in range(100, 201)],
]


class GemmaTokenizerExpander:
    def __init__(
        self,
        student_tokenizer_id: str,
        target_vocab_size: int,
        datasets_dicts: List[dict],
        teacher_tokenizer_id: str = DEFAULT_REF_MODEL_ID,
        ref_model_id: str = DEFAULT_REF_MODEL_ID,
        ref_model_path: Optional[str] = None,
    ):
        self.teacher_tokenizer = GemmaTokenizerFast.from_pretrained(teacher_tokenizer_id)
        self.student_tokenizer = GemmaTokenizerFast.from_pretrained(student_tokenizer_id)
        self.target_vocab_size = target_vocab_size - len(LAST_ADDED_TOKENS)
        self.datasets_dicts = datasets_dicts

        # Load reference SentencePiece model
        self.ref_model = sp_model.ModelProto()
        if ref_model_path is not None:
            # Use local path if provided
            model_path = ref_model_path
        else:
            # Download from HuggingFace Hub
            model_path = hf_hub_download(repo_id=ref_model_id, filename="tokenizer.model")

        with open(model_path, "rb") as f:
            self.ref_model.ParseFromString(f.read())

    def __str__(self):
        return f"TokenizerExpander(teacher_tokenizer={self.teacher_tokenizer.name_or_path}, student_tokenizer={self.student_tokenizer.name_or_path}, target_vocab_size={self.target_vocab_size}, datasets_dicts={self.datasets_dicts})"

    def __repr__(self):
        return self.__str__()

    def _load_datasets(self):
        for dataset_dict in tqdm(self.datasets_dicts, desc="Loading datasets"):
            if dataset_dict.get("limit") is not None:
                dataset = load_dataset(
                    dataset_dict["id"],
                    dataset_dict["subset"],
                    split=dataset_dict["split"],
                    streaming=True,
                )
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
                text = " ".join(
                    row[column] if row[column] is not None else "" for column in columns
                )
                tokens = self.teacher_tokenizer.tokenize(text)
                for token in tokens:
                    freq_dict[token] = freq_dict.get(token, 0) + 1
                    if freq_dict[token] >= dataset_dict["min_freq"]:
                        selected_tokens.add(token)

        return selected_tokens

    def _create_new_tokens_list(self):
        student_vocab = self.student_tokenizer.get_vocab()
        removed_tokens = []
        for token in FIRST_ADDED_TOKENS:
            if token in student_vocab:
                del student_vocab[token]
                removed_tokens.append(token)
        for token in LAST_ADDED_TOKENS:
            if token in student_vocab:
                del student_vocab[token]
                removed_tokens.append(token)
        # sort by their ids
        new_tokens = sorted(student_vocab.items(), key=lambda x: x[1])
        new_tokens = [token for token, id in new_tokens]

        new_tokens = FIRST_ADDED_TOKENS + new_tokens
        student_vocab = {}
        new_vocab_size = len(new_tokens)

        for i in range(new_vocab_size):
            student_vocab[new_tokens[i]] = i

        teacher_vocab = self.teacher_tokenizer.get_vocab()
        teacher_vocab_size = len(teacher_vocab)

        first_added_len = len(FIRST_ADDED_TOKENS)

        # first add 1 letter tokens
        for token, id in tqdm(teacher_vocab.items(), desc="Adding 1-letter tokens"):
            if new_vocab_size >= self.target_vocab_size:
                break
            if len(token) == 1 and token not in student_vocab:
                new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                if new_token_id < first_added_len:
                    new_token_id += first_added_len
                new_tokens.insert(new_token_id, token)
                new_vocab_size += 1
                student_vocab[token] = new_vocab_size

        # second add 2 letter tokens
        for token, id in tqdm(teacher_vocab.items(), desc="Adding 2-letter tokens"):
            if new_vocab_size >= self.target_vocab_size:
                break
            if len(token) == 2 and token not in student_vocab:
                new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                if new_token_id < first_added_len:
                    new_token_id += first_added_len
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
                if new_token_id < first_added_len:
                    new_token_id += first_added_len
                new_tokens.insert(new_token_id, token)
                new_vocab_size += 1
                student_vocab[token] = new_vocab_size

        # last add remaining 3,4,5 gradually letter tokens until reacheing target vocab size
        for i in range(3, 6):
            if new_vocab_size >= self.target_vocab_size:
                break
            for token, id in tqdm(teacher_vocab.items(), desc=f"Adding {i}-letter tokens"):
                if len(token) == i and token not in student_vocab:
                    new_token_id = round(id * (new_vocab_size / teacher_vocab_size))
                    if new_token_id < first_added_len:
                        new_token_id += first_added_len
                    new_tokens.insert(new_token_id, token)
                    new_vocab_size += 1
        # remove last tokens if new_vocab_size >= self.target_vocab_size:
        # be sure there is not duplicated items in the list
        student_vocab = {}
        for i in range(self.target_vocab_size):
            student_vocab[new_tokens[i]] = i
        new_tokens = list(student_vocab.keys())
        new_tokens.extend(LAST_ADDED_TOKENS)
        new_tokens.append("<image_soft_token>")

        print("Distinct new tokens length:", len(set(new_tokens)))

        return new_tokens

    def _get_piece_type(self, token: str) -> int:
        if token == "<unk>":
            return 2  # UNKNOWN type
        if token in {"<pad>", "<eos>", "<bos>"}:
            return 3  # CONTROL type
        if token.startswith("<0x"):
            return 6  # BYTE type
        if token == "[multimodal]" or (token.startswith("<") and token.endswith(">")):
            return 4  # USER_DEFINED type

        return 1  # NORMAL type

    def create_new_tokenizer(self):
        new_tokens = self._create_new_tokens_list()

        model = sp_model.ModelProto()
        model.CopyFrom(self.ref_model)

        del model.pieces[:]

        normal_token_score = -0
        for token in tqdm(new_tokens, desc="Creating sentencepiece model"):
            piece = model.pieces.add()
            piece.piece = token
            piece_type = self._get_piece_type(token)
            if piece_type > 1:
                piece.type = piece_type

            # Special tokens and control tokens have score 0
            if piece_type > 1:  # UNKNOWN, CONTROL, USER_DEFINED, BYTE
                if piece_type == 6:  # BYTE type
                    piece.score = -100 * 100 * 100  # Very low score for byte tokens
                else:
                    piece.score = 0
            else:
                piece.score = normal_token_score

            normal_token_score -= 1

        """ model.trainer_spec.unk_id = 3
        model.trainer_spec.bos_id = 2
        model.trainer_spec.eos_id = 1
        model.trainer_spec.pad_id = 0

        model.trainer_spec.eos_piece = "<eos>"
        model.trainer_spec.bos_piece = "<bos>" """

        model.trainer_spec.vocab_size = len(new_tokens)
        print(model.trainer_spec.vocab_size)

        with open("new_tokenizer.model", "wb") as f:
            f.write(model.SerializeToString())

        return "new_tokenizer.model"
