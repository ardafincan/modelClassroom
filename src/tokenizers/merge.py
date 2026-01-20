class Merge():
    def __init__(self, index: int, subtokens: list):
        self.index = index
        self.subtokens = subtokens

def serializeMerges(merges: list): 
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

#below functions will be implemented later
def index_merges(mergeDict: dict):
    mergeSet = {}
    for merge in mergeDict:
        mergeSet.add()
        

def handleMerges(source: Tokenizer, target: Tokenizer, token: str): 
    source_merges = source.merges
    target_merges = target.merges
    return