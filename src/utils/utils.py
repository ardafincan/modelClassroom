from ..tokenizers.tokenizer import Tokenizer
from sentencepiece import sentencepiece_model_pb2 as spmodel 

def get_nested(dictionary: dict, keys: list[str]):
    """Utility function to get right value in nested dictionaries with given keys.
    
    Args: 
        dictionary (dict): Dictionary to get values from.
        keys (list[str]): Keys to get values from Dictionary.
    Returns: 
        corresponding value of the given keys in given nested dictionary.
    """
    result = dictionary
    for key in keys: 
        result = result[key] 
    return result

def parseTokenizerModelFile(path: str, target_path: str = ""):
    model = spmodel.ModelProto()
    with open(path, 'rb') as f:
        file_content = f.read()
        print(f"File size: {len(file_content)} bytes")
        # convert to string from bytes
        model.ParseFromString(file_content)
        
    if not (target_path == ""):
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(str(model))

    return model

def getTokenizerFromModel(model: spmodel.ModelProto): 
    vocabDict: None
    return 0
