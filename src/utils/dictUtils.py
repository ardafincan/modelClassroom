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