from datasets import load_dataset
from tqdm import tqdm
import json

ds1_test = []
ds2_test = []
ds3_test = []

def return_freq_dict(tokenizer, texts, filename):
    freq = {}
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
    # order by frequency
    freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
    
    # Save to JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(freq, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filename}")
    
    return freq



def return_selected_token_array(dict, min_freq,test_array):
    array = []
    for token, freq in dict.items():
        if freq >= min_freq:
            test_array.append(token)
            array.append(token)
    print(len(array))
    return array


def func(tokenizer):
    ds1 = load_dataset("Ba2han/TDK_Sozluk-Turkish-v2")
    ds2 = load_dataset("alibayram/cosmos-corpus-00-5")
    ds3 = load_dataset("alibayram/wikiset")

    final_set = set()

    # First dataset
    ds1_array=[]

    for item in ds1['train']:
        text = (item['madde']) + " " + (item['anlam']) + " " + (item['ornek'] if item['ornek'] else '')
        ds1_array.append(text)

    dict1 = return_freq_dict(tokenizer, ds1_array,"tdk.json")

    # Second dataset
    data2 = ds2['train']['text']
    dict2 = return_freq_dict(tokenizer, data2,"cosmos.json")

    # Third dataset
    data3 = ds3['train'].filter(lambda x: x["lang"] == "tr")
    data3 = data3['text']
    dict3 = return_freq_dict(tokenizer, data3,"wikiset.json")

    final_set.update(return_selected_token_array(dict1, 3,ds1_test))
    final_set.update(return_selected_token_array(dict2, 1600,ds2_test))
    final_set.update(return_selected_token_array(dict3, 180,ds3_test))

    return list(final_set)