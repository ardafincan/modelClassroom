"""
Gemma Processor Builder

Creates a Gemma-compatible processor from a custom SentencePiece tokenizer.model file.
Generates all necessary files to be used with Gemma3Processor and optionally pushes to HuggingFace Hub.

Output files (matching google/gemma-3-4b-it structure):
- tokenizer.model
- tokenizer.json
- tokenizer_config.json
- added_tokens.json
- special_tokens_map.json
- preprocessor_config.json
- processor_config.json
- chat_template.json
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download, HfApi
from transformers import GemmaTokenizerFast, Gemma3Processor


# Core special tokens with their fixed IDs (positions 0-5)
CORE_SPECIAL_TOKENS: Dict[str, int] = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "[multimodal]": 5,
}

# Turn and HTML special tokens with their fixed IDs
FIXED_SPECIAL_TOKENS: Dict[str, int] = {
    # Turn tokens
    "<start_of_turn>": 105,
    "<end_of_turn>": 106,
    # Table tokens
    "<table>": 110,
    "<caption>": 111,
    "<thead>": 112,
    "<tbody>": 113,
    "<tfoot>": 114,
    "<tr>": 115,
    "<th>": 116,
    "<td>": 117,
    "</table>": 118,
    "</caption>": 119,
    "</thead>": 120,
    "</tbody>": 121,
    "</tfoot>": 122,
    "</tr>": 123,
    "</th>": 124,
    "</td>": 125,
    # Heading tokens
    "<h1>": 126,
    "<h2>": 127,
    "<h3>": 128,
    "<h4>": 129,
    "<h5>": 130,
    "<h6>": 131,
    "<blockquote>": 132,
    "</h1>": 133,
    "</h2>": 134,
    "</h3>": 135,
    "</h4>": 136,
    "</h5>": 137,
    "</h6>": 138,
    "</blockquote>": 139,
    # Formatting tokens
    "<strong>": 140,
    "<em>": 141,
    "<b>": 142,
    "<i>": 143,
    "<u>": 144,
    "<s>": 145,
    "<sub>": 146,
    "<sup>": 147,
    "<code>": 148,
    "</strong>": 149,
    "</em>": 150,
    "</b>": 151,
    "</i>": 152,
    "</u>": 153,
    "</s>": 154,
    "</sub>": 155,
    "</sup>": 156,
    "</code>": 157,
    # More HTML tokens
    "<a>": 158,
    "<html>": 159,
    "<body>": 160,
    "<img>": 161,
    "<span>": 162,
    "<bbox>": 163,
    "<ul>": 164,
    "<li>": 165,
    "<div>": 166,
    "<iframe>": 167,
    "<footer>": 168,
    "</a>": 169,
    "</html>": 170,
    "</body>": 171,
    "</img>": 172,
    "</span>": 173,
    "</bbox>": 174,
    "</ul>": 175,
    "</li>": 176,
    "</div>": 177,
    "</iframe>": 178,
    "</footer>": 179,
    # Image tokens
    "<start_of_image>": 130999,
    "<end_of_image>": 131000,
}

# Special bracket tokens
BRACKET_TOKENS = {"[]", "<>", "<?>", "[,]", "<=>", "[-]", "[:]", "[*]", "[…]"}

# Image soft token is the last token
IMAGE_SOFT_TOKEN = "<image_soft_token>"
IMAGE_SOFT_TOKEN_ID = 131072


class GemmaProcessorBuilder:
    """
    Builds a Gemma-compatible processor from a custom tokenizer.model file.
    
    Usage:
        builder = GemmaProcessorBuilder(
            tokenizer_model_path="new_tokenizer.model",
            output_dir="./custom-gemma-processor"
        )
        builder.build()
        
        # Optionally push to HuggingFace Hub
        builder.push_to_hub("your-username/your-model-name")
    """
    
    def __init__(
        self,
        tokenizer_model_path: str,
        output_dir: str,
        reference_processor_id: str = "google/gemma-3-4b-it",
    ):
        """
        Initialize the processor builder.
        
        Args:
            tokenizer_model_path: Path to the custom tokenizer.model file
            output_dir: Directory to save the generated processor files
            reference_processor_id: HuggingFace model ID to use as reference for configs
        """
        self.tokenizer_model_path = Path(tokenizer_model_path)
        self.output_dir = Path(output_dir)
        self.reference_processor_id = reference_processor_id
        
        # Load reference processor for config templates
        self.reference_processor = Gemma3Processor.from_pretrained(reference_processor_id)
        
    def build(self) -> str:
        """
        Build the processor and save to output directory.
        
        Returns:
            Path to the output directory
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Copy tokenizer.model to output directory
        print("Step 1: Copying tokenizer.model...")
        shutil.copy(self.tokenizer_model_path, self.output_dir / "tokenizer.model")
        
        # Step 2: Load and convert tokenizer to get tokenizer.json
        print("Step 2: Loading and converting tokenizer...")
        tokenizer_json = self._load_and_convert_tokenizer()
        
        # Step 3: Build added_tokens
        print("Step 3: Building added_tokens...")
        added_tokens_list = self._build_added_tokens(tokenizer_json)
        tokenizer_json["added_tokens"] = added_tokens_list
        
        # Step 4: Save tokenizer.json
        print("Step 4: Saving tokenizer.json...")
        with open(self.output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        
        # Step 5: Save added_tokens.json (dict format for HF compatibility)
        print("Step 5: Saving added_tokens.json...")
        added_tokens_dict = {t["content"]: t["id"] for t in added_tokens_list}
        with open(self.output_dir / "added_tokens.json", "w", encoding="utf-8") as f:
            json.dump(added_tokens_dict, f, indent=2, ensure_ascii=False)
        
        # Step 6: Build and save tokenizer_config.json
        print("Step 6: Saving tokenizer_config.json...")
        tokenizer_config = self._build_tokenizer_config(tokenizer_json)
        with open(self.output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        # Step 7: Build and save special_tokens_map.json
        print("Step 7: Saving special_tokens_map.json...")
        special_tokens_map = self._build_special_tokens_map()
        with open(self.output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
        
        # Step 8: Copy processor files from reference
        print("Step 8: Copying processor configuration files...")
        self._copy_processor_files()
        
        # Step 9: Verify the processor works
        print("Step 9: Verifying processor...")
        self._verify_processor()
        
        print(f"\n✓ Processor successfully created at: {self.output_dir}")
        self._print_file_list()
        return str(self.output_dir)
    
    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: str = "Upload Gemma processor files"
    ) -> str:
        """
        Push all processor files to HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/model-name")
            private: Whether to create a private repository
            token: HuggingFace token (uses cached token if not provided)
            commit_message: Commit message for the upload
            
        Returns:
            URL of the repository
        """
        print(f"\nPushing to HuggingFace Hub: {repo_id}")
        
        api = HfApi(token=token)
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")
        
        # Upload all files in output directory
        files = list(self.output_dir.iterdir())
        for file_path in files:
            if file_path.is_file():
                print(f"  Uploading {file_path.name}...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    commit_message=f"{commit_message}: {file_path.name}"
                )
        
        repo_url = f"https://huggingface.co/{repo_id}"
        print(f"\n✓ Successfully pushed to: {repo_url}")
        return repo_url
    
    def _print_file_list(self):
        """Print list of generated files."""
        print("\nGenerated files:")
        files = sorted(self.output_dir.iterdir())
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.2f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.2f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"  - {f.name} ({size_str})")
    
    def _load_and_convert_tokenizer(self) -> dict:
        """Load SentencePiece model and convert to tokenizer.json format."""
        # Create a temporary directory for the tokenizer
        temp_dir = self.output_dir / "_temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy the tokenizer.model to temp directory
            shutil.copy(self.tokenizer_model_path, temp_dir / "tokenizer.model")
            
            # Build a minimal tokenizer_config.json for loading
            minimal_config = {
                "tokenizer_class": "GemmaTokenizer",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
            with open(temp_dir / "tokenizer_config.json", "w") as f:
                json.dump(minimal_config, f)
            
            # Load using GemmaTokenizerFast
            tokenizer = GemmaTokenizerFast.from_pretrained(
                str(temp_dir),
                vocab_file=str(temp_dir / "tokenizer.model")
            )
            
            # Save to get tokenizer.json
            tokenizer.save_pretrained(str(temp_dir))
            
            # Load the generated tokenizer.json
            with open(temp_dir / "tokenizer.json", "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
            
            # Copy post_processor from reference tokenizer.json (includes BOS token handling)
            ref_tokenizer_path = hf_hub_download(
                repo_id=self.reference_processor_id,
                filename="tokenizer.json"
            )
            with open(ref_tokenizer_path, "r", encoding="utf-8") as f:
                ref_tokenizer_json = json.load(f)
            
            # Copy post_processor to add BOS token on encode
            if "post_processor" in ref_tokenizer_json:
                tokenizer_json["post_processor"] = ref_tokenizer_json["post_processor"]
            
            return tokenizer_json
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _build_added_tokens(self, tokenizer_json: dict) -> List[Dict]:
        """Build added_tokens list from the vocab, marking special tokens correctly."""
        vocab = tokenizer_json.get("model", {}).get("vocab", {})
        added_tokens = []
        
        for token, token_id in vocab.items():
            if self._is_special_token(token):
                added_tokens.append(self._create_added_token_entry(token, token_id))
        
        return sorted(added_tokens, key=lambda x: x["id"])
    
    def _create_added_token_entry(self, token: str, token_id: int) -> Dict:
        """Create an added_token entry for a token."""
        return {
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        }
    
    def _is_special_token(self, token: str) -> bool:
        """Check if a token should be marked as a special token."""
        # Core special tokens
        if token in CORE_SPECIAL_TOKENS:
            return True
        
        # Fixed special tokens (turn, HTML, image tokens)
        if token in FIXED_SPECIAL_TOKENS:
            return True
        
        # Image soft token
        if token == IMAGE_SOFT_TOKEN:
            return True
        
        # Unused tokens
        if token.startswith("<unused"):
            return True
        
        # Byte tokens (<0x00> through <0xFF>)
        if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
            return True
        
        # Special bracket tokens
        if token in BRACKET_TOKENS:
            return True
        
        return False
    
    def _build_tokenizer_config(self, tokenizer_json: dict) -> Dict:
        """Build tokenizer_config.json with correct token mappings."""
        config = {
            "additional_special_tokens": None,
            "backend": "tokenizers",
            "boi_token": "<start_of_image>",
            "bos_token": "<bos>",
            "clean_up_tokenization_spaces": False,
            "eoi_token": "<end_of_image>",
            "eos_token": "<eos>",
            "image_token": IMAGE_SOFT_TOKEN,
            "is_local": False,
            "mask_token": "<mask>",
            "model_max_length": 1000000000000000019884624838656,
            "model_specific_special_tokens": {
                "boi_token": "<start_of_image>",
                "eoi_token": "<end_of_image>",
                "image_token": IMAGE_SOFT_TOKEN
            },
            "pad_token": "<pad>",
            "processor_class": "Gemma3Processor",
            "sp_model_kwargs": None,
            "spaces_between_special_tokens": False,
            "tokenizer_class": "GemmaTokenizer",
            "unk_token": "<unk>",
            "use_default_system_prompt": False,
            "vocab_file": "tokenizer.model"
        }
        
        return config
    
    def _build_special_tokens_map(self) -> Dict:
        """Build special_tokens_map.json."""
        return {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "mask_token": "<mask>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        }
    
    def _copy_processor_files(self):
        """Copy processor_config.json, chat_template, and save image processor config from reference."""
        # Download and copy processor_config.json
        processor_config_path = hf_hub_download(
            repo_id=self.reference_processor_id,
            filename="processor_config.json"
        )
        shutil.copy(processor_config_path, self.output_dir / "processor_config.json")
        
        # Save the image processor config (creates preprocessor_config.json)
        self.reference_processor.image_processor.save_pretrained(str(self.output_dir))
        
        # Download chat_template and save both .json and .jinja formats
        try:
            chat_template_path = hf_hub_download(
                repo_id=self.reference_processor_id,
                filename="chat_template.json"
            )
            shutil.copy(chat_template_path, self.output_dir / "chat_template.json")
            
            # Also extract and save as .jinja (raw template format)
            with open(chat_template_path, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
            if "chat_template" in chat_data:
                with open(self.output_dir / "chat_template.jinja", "w", encoding="utf-8") as f:
                    f.write(chat_data["chat_template"])
        except Exception:
            # Try chat_template.jinja directly
            try:
                chat_template_path = hf_hub_download(
                    repo_id=self.reference_processor_id,
                    filename="chat_template.jinja"
                )
                shutil.copy(chat_template_path, self.output_dir / "chat_template.jinja")
            except Exception:
                print("  Note: No chat template found in reference, skipping...")
    
    def _verify_processor(self):
        """Verify the generated processor loads correctly."""
        try:
            processor = Gemma3Processor.from_pretrained(str(self.output_dir))
            tokenizer = processor.tokenizer
            
            print(f"  Vocab size: {tokenizer.vocab_size}")
            print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
            print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
            print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
            print(f"  UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")
            
            # Test tokenization
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            print(f"  Test encode/decode: '{test_text}' -> {tokens} -> '{decoded}'")
            
        except Exception as e:
            print(f"  Warning: Verification failed: {e}")
            print("  The processor files were created but may need manual adjustment.")


def main():
    """CLI interface for the processor builder."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build a Gemma processor from a custom tokenizer.model"
    )
    parser.add_argument(
        "tokenizer_model",
        help="Path to the tokenizer.model file"
    )
    parser.add_argument(
        "-o", "--output",
        default="./custom-gemma-processor",
        help="Output directory for the processor (default: ./custom-gemma-processor)"
    )
    parser.add_argument(
        "-r", "--reference",
        default="google/gemma-3-4b-it",
        help="Reference processor model ID (default: google/gemma-3-4b-it)"
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Push to HuggingFace Hub (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository when pushing to Hub"
    )
    
    args = parser.parse_args()
    
    builder = GemmaProcessorBuilder(
        tokenizer_model_path=args.tokenizer_model,
        output_dir=args.output,
        reference_processor_id=args.reference,
    )
    builder.build()
    
    # Push to Hub if requested
    if args.push_to_hub:
        builder.push_to_hub(
            repo_id=args.push_to_hub,
            private=args.private
        )


if __name__ == "__main__":
    main()
