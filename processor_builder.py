import os
import json
import shutil

from huggingface_hub import HfApi, hf_hub_download

from transformers import GemmaTokenizerFast, Gemma3Processor

class ProcessorBuilder:
    def __init__(
        self, student_tokenizer_model_path: str, 
        teacher_processor: str = "google/gemma-3-4b-it"
    ):
        student_tokenizer = GemmaTokenizerFast.from_pretrained(
            ".", vocab_file=student_tokenizer_model_path
        )
        student_tokenizer.save_pretrained("student_tokenizer")

        with open("student_tokenizer/tokenizer_config.json", "r") as f:
            self.student_tokenizer_config = json.load(f)

        with open("student_tokenizer/tokenizer.json", "r") as f:
            self.student_tokenizer_json = json.load(f)

        # move model file to student_tokenizer directory
        shutil.copyfile(
            student_tokenizer_model_path, 
            f"student_tokenizer/tokenizer.model"
        )
        api = HfApi()
        model_info = api.model_info(teacher_processor)
        files = [file.rfilename for file in model_info.siblings]

        # needed files [added_tokens.json, chat_template.json, preprocessor_config.json,
        # processor_config.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json]
        # download files
        needed_files = [
            "added_tokens.json",
            "chat_template.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        local_files = {}
        self.local_dir = f"./{teacher_processor.replace('/', '_')}_files"
        for file in needed_files:
            if file in files:
                local_path = hf_hub_download(
                    repo_id=teacher_processor,
                    filename=file,
                    local_dir=self.local_dir,
                )
                local_files[file] = local_path

    def build(self):
        print("Building new processor with student tokenizer...")

        print("Merging tokenizer.json files...")
        with open(f"{self.local_dir}/tokenizer.json", "r") as f:
            gemma_tokenizer_json = json.loads(f.read())

        with open("student_tokenizer/tokenizer.json", "r") as f:
            tokenizer_json = json.loads(f.read())
            added_tokens = tokenizer_json["added_tokens"]
            added_tokens.append(
                {
                    "id": len(tokenizer_json["model"]["vocab"]),
                    "content": "<image_soft_token>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            )
            vocab = tokenizer_json["model"]["vocab"]
            merges = tokenizer_json["model"].get("merges", [])

        gemma_tokenizer_json["added_tokens"] = added_tokens
        gemma_tokenizer_json["model"]["vocab"] = vocab
        gemma_tokenizer_json["model"]["merges"] = merges

        with open("student_tokenizer/tokenizer.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(gemma_tokenizer_json, ensure_ascii=False))

        print("Merging tokenizer_config.json files...")
        with open(f"{self.local_dir}/tokenizer_config.json", "r") as f:
            gemma_tokenizer_config = json.load(f)
            gemma_tokenizer_config["added_tokens_decoder"] = {}

        for token in added_tokens:
            token_id = str(token["id"])
            del token["id"]
            gemma_tokenizer_config["added_tokens_decoder"][token_id] = token

        with open("student_tokenizer/tokenizer_config.json", "w", encoding="utf-8") as f:
            gemma_tokenizer_config["vocab_file"] = "tokenizer.model"
            f.write(json.dumps(gemma_tokenizer_config, ensure_ascii=False))

        print("Extra files copying...")
        with open(f"{self.local_dir}/added_tokens.json", "r") as f:
            added_tokens_data = json.load(f)
            added_tokens_data["<image_soft_token>"] = len(vocab) # new token id
            print(added_tokens_data)

        with open("student_tokenizer/added_tokens.json", "w", encoding="utf-8") as f:
            print("Added <image_soft_token> to added_tokens.json")
            f.write(json.dumps(added_tokens_data, ensure_ascii=False))

        shutil.copyfile(
            f"{self.local_dir}/special_tokens_map.json",
            "student_tokenizer/special_tokens_map.json"
        )

        shutil.copyfile(
            f"{self.local_dir}/chat_template.json",
            "student_tokenizer/chat_template.json"
        )

        with open("student_tokenizer/chat_template.json", "r") as f:
            chat_template = json.load(f)
            template = chat_template["chat_template"]

        # create chat_template.jinja
        with open("student_tokenizer/chat_template.jinja", "w", encoding="utf-8") as f:
            f.write(template)

        shutil.copyfile(
            f"{self.local_dir}/preprocessor_config.json",
            "student_tokenizer/preprocessor_config.json"
        )

        shutil.copyfile(
            f"{self.local_dir}/processor_config.json",
            "student_tokenizer/processor_config.json"
        )

        print("Processor build complete. New processor is in 'student_tokenizer' directory.")
        # push files to huggingface
        api = HfApi()
        api.create_repo(
            repo_id="alibayram/tr-gemma-128k-processor",
        )

        for file in os.listdir("student_tokenizer"):
            api.upload_file(
                repo_id="alibayram/tr-gemma-128k-processor",
                path_or_fileobj=f"student_tokenizer/{file}",
                path_in_repo=file,
            )

if __name__ == "__main__":
    builder = ProcessorBuilder("student_tokenizer/tokenizer.model")
    builder.build()
