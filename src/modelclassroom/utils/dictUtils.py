def get_nested(dictionary: dict, keys: list[str]):
    result = dictionary
    for key in keys: 
        result = result[key] 
    return result