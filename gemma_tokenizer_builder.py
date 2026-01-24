"""
Gemma Tokenizer Builder

A unified tool to create a custom Gemma tokenizer/processor from scratch.
Takes datasets, student tokenizer, and target vocab size, then:
1. Expands vocabulary based on dataset frequency analysis
2. Creates SentencePiece tokenizer.model with proper special token positions
3. Generates all processor files (tokenizer.json, configs, chat template)
4. Optionally pushes everything to HuggingFace Hub

Usage:
    python gemma_tokenizer_builder.py \
        --student-tokenizer alibayram/turkish-tokenizer \
        --target-vocab-size 131073 \
        --output-dir ./my-processor \
        --datasets "alibayram/turkish-wiki:text" "alibayram/news:content" \
        --push-to-hub username/my-tokenizer
"""

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_model
from tqdm import tqdm
from transformers import GemmaTokenizerFast, Gemma3Processor


# =============================================================================
# Token Configuration
# =============================================================================

CORE_SPECIAL_TOKENS = {
    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "[multimodal]": 5,
}

FIXED_SPECIAL_TOKENS = {
    # Turn tokens (105-106)
    "<start_of_turn>": 105, "<end_of_turn>": 106,
    # Table tokens (110-125)
    "<table>": 110, "<caption>": 111, "<thead>": 112, "<tbody>": 113,
    "<tfoot>": 114, "<tr>": 115, "<th>": 116, "<td>": 117,
    "</table>": 118, "</caption>": 119, "</thead>": 120, "</tbody>": 121,
    "</tfoot>": 122, "</tr>": 123, "</th>": 124, "</td>": 125,
    # Heading tokens (126-139)
    "<h1>": 126, "<h2>": 127, "<h3>": 128, "<h4>": 129, "<h5>": 130, "<h6>": 131,
    "<blockquote>": 132, "</h1>": 133, "</h2>": 134, "</h3>": 135,
    "</h4>": 136, "</h5>": 137, "</h6>": 138, "</blockquote>": 139,
    # Formatting tokens (140-157)
    "<strong>": 140, "<em>": 141, "<b>": 142, "<i>": 143, "<u>": 144,
    "<s>": 145, "<sub>": 146, "<sup>": 147, "<code>": 148,
    "</strong>": 149, "</em>": 150, "</b>": 151, "</i>": 152, "</u>": 153,
    "</s>": 154, "</sub>": 155, "</sup>": 156, "</code>": 157,
    # More HTML tokens (158-179)
    "<a>": 158, "<html>": 159, "<body>": 160, "<img>": 161, "<span>": 162,
    "<bbox>": 163, "<ul>": 164, "<li>": 165, "<div>": 166, "<iframe>": 167,
    "<footer>": 168, "</a>": 169, "</html>": 170, "</body>": 171, "</img>": 172,
    "</span>": 173, "</bbox>": 174, "</ul>": 175, "</li>": 176, "</div>": 177,
    "</iframe>": 178, "</footer>": 179,
    # Image tokens
    "<start_of_image>": 130999, "<end_of_image>": 131000,
}

BRACKET_TOKENS = {"[]", "<>", "<?>", "[,]", "<=>", "[-]", "[:]", "[*]", "[…]"}
IMAGE_SOFT_TOKEN = "<image_soft_token>"
IMAGE_SOFT_TOKEN_ID = 131072
BYTE_TOKENS_START = 180
DEFAULT_REF_MODEL = "google/gemma-3-4b-it"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset to use for vocabulary expansion."""
    name: str
    columns: List[str]  # Can specify multiple columns
    split: str = "train"
    streaming: bool = True
    max_samples: Optional[int] = None
    min_freq: int = 1  # Minimum frequency threshold for tokens from this dataset


# Default Turkish datasets for vocabulary expansion
DEFAULT_DATASETS = [
    DatasetConfig(
        name="Ba2han/TDK_Sozluk-Turkish-v2",
        columns=["madde", "anlam", "ornek"],
        split="train",
        min_freq=3,
    ),
    DatasetConfig(
        name="alibayram/cosmos-corpus-00-5",
        columns=["text"],
        split="train",
        min_freq=1600,
    ),
]


@dataclass
class BuildConfig:
    """Configuration for the tokenizer build process."""
    student_tokenizer_id: str
    target_vocab_size: int
    output_dir: Path
    datasets: List[DatasetConfig]
    reference_model_id: str = DEFAULT_REF_MODEL
    push_to_hub: Optional[str] = None
    private: bool = False
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


# =============================================================================
# Main Builder Class
# =============================================================================

class GemmaTokenizerBuilder:
    """
    Builds a complete Gemma-compatible tokenizer/processor from a student tokenizer.
    
    Example:
        builder = GemmaTokenizerBuilder(
            student_tokenizer_id="alibayram/turkish-tokenizer",
            target_vocab_size=131073,
            output_dir="./my-processor",
            datasets=[DatasetConfig("alibayram/wiki", "text")]
        )
        builder.build()
        builder.push_to_hub("username/my-tokenizer")
    """
    
    def __init__(
        self,
        student_tokenizer_id: str,
        target_vocab_size: int,
        output_dir: str,
        datasets: List[DatasetConfig],
        reference_model_id: str = DEFAULT_REF_MODEL,
    ):
        self.config = BuildConfig(
            student_tokenizer_id=student_tokenizer_id,
            target_vocab_size=target_vocab_size,
            output_dir=Path(output_dir),
            datasets=datasets,
            reference_model_id=reference_model_id,
        )
        
        # Load tokenizers
        print("Loading tokenizers...")
        self.teacher_tokenizer = GemmaTokenizerFast.from_pretrained(reference_model_id)
        self.student_tokenizer = GemmaTokenizerFast.from_pretrained(student_tokenizer_id)
        
        # Load reference SentencePiece model
        model_path = hf_hub_download(repo_id=reference_model_id, filename="tokenizer.model")
        self.ref_sp_model = sp_model.ModelProto()
        with open(model_path, 'rb') as f:
            self.ref_sp_model.ParseFromString(f.read())
        
        # Build special token sets
        self._all_special = set(CORE_SPECIAL_TOKENS.keys())
        self._all_special.update(FIXED_SPECIAL_TOKENS.keys())
        self._all_special.add(IMAGE_SOFT_TOKEN)
        self._all_special.update(BRACKET_TOKENS)
        self._byte_tokens = {f"<0x{i:02X}>" for i in range(256)}
    
    def build(self) -> Path:
        """Build the tokenizer and processor, return output directory path."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("GEMMA TOKENIZER BUILDER")
        print("="*60)
        
        # Step 1: Analyze datasets and get token frequencies
        print("\n[1/6] Analyzing datasets for token frequencies...")
        token_freqs = self._analyze_datasets()
        
        # Step 2: Build expanded vocabulary
        print("\n[2/6] Building expanded vocabulary...")
        vocab_tokens = self._build_vocabulary(token_freqs)
        
        # Step 3: Create SentencePiece model
        print("\n[3/6] Creating tokenizer.model...")
        self._create_sp_model(vocab_tokens)
        
        # Step 4: Generate processor files
        print("\n[4/6] Generating processor files...")
        self._create_processor_files()
        
        # Step 5: Verify
        print("\n[5/6] Verifying processor...")
        self._verify()
        
        # Step 6: Summary
        print("\n[6/6] Done!")
        self._print_summary()
        
        return self.config.output_dir
    
    def push_to_hub(self, repo_id: str, private: bool = False) -> str:
        """Push all files to HuggingFace Hub."""
        print(f"\nPushing to HuggingFace Hub: {repo_id}")
        
        api = HfApi()
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        
        files = list(self.config.output_dir.iterdir())
        for f in tqdm(files, desc="Uploading files"):
            if f.is_file():
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=repo_id,
                )
        
        url = f"https://huggingface.co/{repo_id}"
        print(f"✓ Pushed to: {url}")
        return url
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _analyze_datasets(self) -> Dict[str, Counter]:
        """Analyze datasets and return token frequency counts per dataset."""
        dataset_freqs = {}
        
        for ds_config in self.config.datasets:
            cols_str = ",".join(ds_config.columns)
            print(f"  Processing {ds_config.name} [{cols_str}] (min_freq={ds_config.min_freq})...")
            
            token_freq = Counter()
            ds = load_dataset(
                ds_config.name,
                split=ds_config.split,
                streaming=ds_config.streaming
            )
            
            samples = ds.take(ds_config.max_samples) if ds_config.max_samples else ds
            
            for sample in tqdm(samples, desc=f"  {ds_config.name}", leave=False):
                for column in ds_config.columns:
                    text = sample.get(column, "")
                    if text:
                        tokens = self.student_tokenizer.tokenize(text)
                        token_freq.update(tokens)
            
            # Apply min_freq filter
            filtered_freq = Counter({
                token: count for token, count in token_freq.items()
                if count >= ds_config.min_freq
            })
            
            dataset_freqs[ds_config.name] = filtered_freq
            print(f"    → {len(token_freq)} unique tokens, {len(filtered_freq)} after min_freq filter")
        
        return dataset_freqs
    
    def _build_vocabulary(self, dataset_freqs: Dict[str, Counter]) -> List[str]:
        """Build vocabulary list using STUDENT tokenizer as the base."""
        # Merge all dataset frequencies
        merged_freq = Counter()
        for freq in dataset_freqs.values():
            merged_freq.update(freq)
        
        print(f"  Total unique tokens after merge: {len(merged_freq)}")
        
        # Get student vocabulary (this is our base)
        student_vocab = self.student_tokenizer.get_vocab()
        
        # Get regular tokens from STUDENT tokenizer, sorted by their original ID
        student_regular = [
            (token, token_id) for token, token_id in student_vocab.items()
            if token not in self._all_special and token not in self._byte_tokens
            and not token.startswith("<unused")
        ]
        # Sort by original token ID to maintain order
        student_regular.sort(key=lambda x: x[1])
        student_tokens = [token for token, _ in student_regular]
        
        # Calculate available slots for regular tokens
        # Reserved: core(6) + unused0-98(99) + turn(2) + unused99-101(3) + html(70) + bytes(256) + image(2) + unused_end + image_soft(1)
        num_reserved = 6 + 99 + 2 + 3 + 70 + 256 + 2 + (IMAGE_SOFT_TOKEN_ID - 131001) + 1
        num_regular_slots = self.config.target_vocab_size - num_reserved
        
        # Use student tokens as base, truncate to fit target size
        final_regular = student_tokens[:num_regular_slots]
        
        print(f"  Student vocab size: {len(student_vocab)}")
        print(f"  Student regular tokens: {len(student_tokens)}")
        print(f"  Available slots: {num_regular_slots}")
        print(f"  Final regular tokens: {len(final_regular)}")
        
        return final_regular
    
    def _create_sp_model(self, regular_tokens: List[str]):
        """Create and save the SentencePiece model."""
        # Build final token list with correct positions
        final_tokens = [""] * (IMAGE_SOFT_TOKEN_ID + 1)
        
        # 1. Core special tokens (0-5)
        for token, idx in CORE_SPECIAL_TOKENS.items():
            final_tokens[idx] = token
        
        # 2. Unused tokens (6-104)
        for i in range(99):
            final_tokens[6 + i] = f"<unused{i}>"
        
        # 3. Fixed special tokens (105-106, 110-179, 130999-131000)
        for token, idx in FIXED_SPECIAL_TOKENS.items():
            final_tokens[idx] = token
        
        # 4. Unused 99-101 (107-109)
        final_tokens[107] = "<unused99>"
        final_tokens[108] = "<unused100>"
        final_tokens[109] = "<unused101>"
        
        # 5. Byte tokens (180-435)
        for i in range(256):
            final_tokens[BYTE_TOKENS_START + i] = f"<0x{i:02X}>"
        
        # 6. Image soft token (last)
        final_tokens[IMAGE_SOFT_TOKEN_ID] = IMAGE_SOFT_TOKEN
        
        # 7. Fill remaining with regular tokens + unused
        regular_idx = 0
        unused_counter = 102
        
        for pos in range(BYTE_TOKENS_START + 256, IMAGE_SOFT_TOKEN_ID):
            if final_tokens[pos] == "":
                if regular_idx < len(regular_tokens):
                    final_tokens[pos] = regular_tokens[regular_idx]
                    regular_idx += 1
                else:
                    final_tokens[pos] = f"<unused{unused_counter}>"
                    unused_counter += 1
        
        # Create SentencePiece model
        new_model = sp_model.ModelProto()
        new_model.CopyFrom(self.ref_sp_model)
        del new_model.pieces[:]
        
        score = 0
        for token in tqdm(final_tokens, desc="  Creating SP model"):
            piece = new_model.pieces.add()
            piece.piece = token
            piece.type = self._get_piece_type(token)
            piece.score = 0 if piece.type in [2, 3, 4, 6] else score
            if piece.type == 1:
                score -= 1
        
        new_model.trainer_spec.vocab_size = len(final_tokens)
        
        # Save
        model_path = self.config.output_dir / "tokenizer.model"
        with open(model_path, "wb") as f:
            f.write(new_model.SerializeToString())
        
        print(f"  Saved tokenizer.model ({len(final_tokens)} tokens)")
    
    def _get_piece_type(self, token: str) -> int:
        """Get SentencePiece type for a token."""
        if token == "<unk>":
            return 2  # UNKNOWN
        if token in self._all_special or token in self._byte_tokens:
            return 3  # CONTROL
        if token.startswith("<unused"):
            return 4  # USER_DEFINED (not CONTROL to avoid slow special token registration)
        return 1  # NORMAL
    
    def _create_processor_files(self):
        """Generate all processor configuration files."""
        # Load the tokenizer we just created
        temp_dir = self.config.output_dir / "_temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            shutil.copy(self.config.output_dir / "tokenizer.model", temp_dir / "tokenizer.model")
            
            # Minimal config for loading
            with open(temp_dir / "tokenizer_config.json", "w") as f:
                json.dump({
                    "tokenizer_class": "GemmaTokenizer",
                    "bos_token": "<bos>", "eos_token": "<eos>",
                    "pad_token": "<pad>", "unk_token": "<unk>",
                }, f)
            
            tokenizer = GemmaTokenizerFast.from_pretrained(
                str(temp_dir), vocab_file=str(temp_dir / "tokenizer.model")
            )
            tokenizer.save_pretrained(str(temp_dir))
            
            # Load generated tokenizer.json
            with open(temp_dir / "tokenizer.json", "r") as f:
                tokenizer_json = json.load(f)
            
            # Copy post_processor from reference tokenizer.json (includes BOS token handling)
            ref_tokenizer_path = hf_hub_download(
                repo_id=self.config.reference_model_id,
                filename="tokenizer.json"
            )
            with open(ref_tokenizer_path, "r") as f:
                ref_tokenizer_json = json.load(f)
            
            if "post_processor" in ref_tokenizer_json:
                tokenizer_json["post_processor"] = ref_tokenizer_json["post_processor"]
            
            # Build added_tokens
            vocab = tokenizer_json.get("model", {}).get("vocab", {})
            added_tokens = [
                {
                    "id": tid, "content": tok, "single_word": False,
                    "lstrip": False, "rstrip": False, "normalized": False, "special": True
                }
                for tok, tid in vocab.items()
                if self._is_special_token(tok)
            ]
            tokenizer_json["added_tokens"] = sorted(added_tokens, key=lambda x: x["id"])
            
            # Save tokenizer.json
            with open(self.config.output_dir / "tokenizer.json", "w") as f:
                json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Save other configs
        self._save_tokenizer_config()
        self._save_added_tokens()
        self._save_special_tokens_map()
        self._copy_reference_files()
    
    def _is_special_token(self, token: str) -> bool:
        """Check if token is a special token."""
        if token in self._all_special:
            return True
        if token.startswith("<unused"):
            return True
        if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
            return True
        return False
    
    def _save_tokenizer_config(self):
        """Save tokenizer_config.json."""
        config = {
            "additional_special_tokens": None,
            "backend": "tokenizers",
            "boi_token": "<start_of_image>",
            "bos_token": "<bos>",
            "clean_up_tokenization_spaces": False,
            "eoi_token": "<end_of_image>",
            "eos_token": "<eos>",
            "image_token": IMAGE_SOFT_TOKEN,
            "mask_token": "<mask>",
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": "<pad>",
            "processor_class": "Gemma3Processor",
            "tokenizer_class": "GemmaTokenizer",
            "unk_token": "<unk>",
            "vocab_file": "tokenizer.model",
        }
        with open(self.config.output_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def _save_added_tokens(self):
        """Save added_tokens.json in dict format."""
        with open(self.config.output_dir / "tokenizer.json", "r") as f:
            data = json.load(f)
        
        added = {t["content"]: t["id"] for t in data.get("added_tokens", [])}
        with open(self.config.output_dir / "added_tokens.json", "w") as f:
            json.dump(added, f, indent=2)
    
    def _save_special_tokens_map(self):
        """Save special_tokens_map.json."""
        with open(self.config.output_dir / "special_tokens_map.json", "w") as f:
            json.dump({
                "bos_token": "<bos>", "eos_token": "<eos>",
                "mask_token": "<mask>", "pad_token": "<pad>", "unk_token": "<unk>"
            }, f, indent=2)
    
    def _copy_reference_files(self):
        """Copy processor configs and chat template from reference model."""
        ref = self.config.reference_model_id
        
        # processor_config.json
        try:
            path = hf_hub_download(repo_id=ref, filename="processor_config.json")
            shutil.copy(path, self.config.output_dir / "processor_config.json")
        except Exception:
            pass
        
        # preprocessor_config.json (image processor)
        try:
            processor = Gemma3Processor.from_pretrained(ref)
            processor.image_processor.save_pretrained(str(self.config.output_dir))
        except Exception:
            pass
        
        # chat_template
        try:
            path = hf_hub_download(repo_id=ref, filename="chat_template.json")
            shutil.copy(path, self.config.output_dir / "chat_template.json")
            with open(path, "r") as f:
                data = json.load(f)
            if "chat_template" in data:
                with open(self.config.output_dir / "chat_template.jinja", "w") as f:
                    f.write(data["chat_template"])
        except Exception:
            pass
    
    def _verify(self):
        """Verify the generated processor loads correctly."""
        try:
            processor = Gemma3Processor.from_pretrained(str(self.config.output_dir))
            tok = processor.tokenizer
            print(f"  ✓ Vocab size: {tok.vocab_size}")
            print(f"  ✓ pad={tok.pad_token_id}, eos={tok.eos_token_id}, bos={tok.bos_token_id}")
            
            # Quick encode/decode test
            test = "Merhaba dünya!"
            encoded = tok.encode(test)
            decoded = tok.decode(encoded)
            print(f"  ✓ Test: '{test}' → {len(encoded)} tokens → '{decoded}'")
        except Exception as e:
            print(f"  ⚠ Verification warning: {e}")
    
    def _print_summary(self):
        """Print summary of generated files."""
        print("\n" + "="*60)
        print("GENERATED FILES:")
        print("="*60)
        
        files = sorted(self.config.output_dir.iterdir())
        total_size = 0
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                size_str = f"{size/1024/1024:.2f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
                print(f"  {f.name:<30} {size_str:>10}")
        
        print("-"*60)
        print(f"  {'Total:':<30} {total_size/1024/1024:.2f}MB")
        print(f"\nOutput directory: {self.config.output_dir.absolute()}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build a custom Gemma tokenizer/processor from datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default Turkish datasets
  python gemma_tokenizer_builder.py \\
      --student-tokenizer alibayram/turkish-tokenizer \\
      --output-dir ./my-processor

  # With custom datasets (format: name:col1,col2:min_freq:max_samples)
  python gemma_tokenizer_builder.py \\
      --student-tokenizer alibayram/turkish-tokenizer \\
      --datasets "my/dataset:text:1" "other/data:col1,col2:100:5000" \\
      --output-dir ./my-processor \\
      --push-to-hub alibayram/my-tokenizer
        """
    )
    
    parser.add_argument(
        "--student-tokenizer", "-s", required=True,
        help="HuggingFace ID of the student tokenizer"
    )
    parser.add_argument(
        "--target-vocab-size", "-v", type=int, default=131073,
        help="Target vocabulary size (default: 131073)"
    )
    parser.add_argument(
        "--datasets", "-d", nargs="*", default=None,
        help="Datasets in format 'name:columns:min_freq:max_samples'. Columns can be comma-separated. If not specified, uses default Turkish datasets."
    )
    parser.add_argument(
        "--output-dir", "-o", default="./custom-gemma-processor",
        help="Output directory (default: ./custom-gemma-processor)"
    )
    parser.add_argument(
        "--reference-model", "-r", default=DEFAULT_REF_MODEL,
        help=f"Reference Gemma model (default: {DEFAULT_REF_MODEL})"
    )
    parser.add_argument(
        "--push-to-hub", "-p",
        help="Push to HuggingFace Hub (e.g., 'username/repo-name')"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create private repository when pushing"
    )
    
    args = parser.parse_args()
    
    # Parse dataset configs or use defaults
    if args.datasets:
        datasets = []
        for ds_str in args.datasets:
            parts = ds_str.split(":")
            if len(parts) < 2:
                parser.error(f"Invalid dataset format: {ds_str}. Use 'name:columns' or 'name:columns:min_freq:max_samples'")
            
            columns = parts[1].split(",")
            min_freq = int(parts[2]) if len(parts) > 2 else 1
            max_samples = int(parts[3]) if len(parts) > 3 else None
            
            ds_config = DatasetConfig(
                name=parts[0],
                columns=columns,
                min_freq=min_freq,
                max_samples=max_samples
            )
            datasets.append(ds_config)
    else:
        print("Using default Turkish datasets...")
        datasets = DEFAULT_DATASETS
    
    # Build
    builder = GemmaTokenizerBuilder(
        student_tokenizer_id=args.student_tokenizer,
        target_vocab_size=args.target_vocab_size,
        output_dir=args.output_dir,
        datasets=datasets,
        reference_model_id=args.reference_model,
    )
    
    builder.build()
    
    if args.push_to_hub:
        builder.push_to_hub(args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()

