from tokenizer import Tokenizer

class Merge():
    def __init__(self, token: str, merges:dict[int, list[str]]):
        self.token = token
        self.merges = merges

def serializeMerges(merges: list): # fix this seri/deseri functions 
        mergeMap = {}
        for idx, merge in enumerate(merges):
            tempMerge = Merge(idx, merge)
            mergeMap[tempMerge.subtokens[0] + tempMerge.subtokens[1]] = tempMerge
        return mergeMap

def deserializeMerges(merges: dict[str, Merge]):
    mergesList = []
    for merge in merges:
        mergesList.append(merges[merge].subtokens)
    return mergesList

def handleMerges(source: Tokenizer, target: Tokenizer, token: str): 
    source_merges = source.merges
    target_merges = target.merges

    tempMerge: Merge

    for idx, merge in enumerate(source_merges):
        if token == merge[0] + merge[1]:
            mergesForThisTokenDict[idx] = merge
    tempMerge = Merge(idx, mergesForThisTokenDict)
    mergeDict[token] = tempMerge

    return