from collections import defaultdict



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """ 
    
    def get_counts(pre_tokens_cnt: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        counts = {}
        for tokens, count in pre_tokens_cnt.items():
            for pair in zip(tokens[:-1], tokens[1:]):
                counts[pair] = counts.get(pair, 0) + count
        return counts
    
    def merge(pre_tokens_cnt, pair_target):
        newTokens = []
        for token, count in pre_tokens_cnt.items():
            i = 0
            new_pre_token = []
            while i < len(token):
                # 检查是否可以合并
                if i < len(token) - 1 and (token[i], token[i + 1]) == pair_target:
                    new_pre_token.append(token[i] + token[i + 1])  # 合并成一个新的 bytes
                    i += 2
                else:
                    new_pre_token.append(token[i])
                    i += 1
            new_pre_token = tuple(new_pre_token)
            newTokens.append((token, new_pre_token, count))
        return newTokens

    # 接下来的改进路径是降低时间复杂度！
    def encode(input_path, vocab_size, special_tokens):
        num_merges = vocab_size - 256 - len(special_tokens)# vocab_size是我们期望的词汇表库的大小，256是Unicode编码上限，还需要给特殊字符留位置
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 + len(special_tokens)")
        
        # print(f"num merges: {num_merges}")
        
        if isinstance(input_path, os.PathLike):
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
                # print("-------------text Length:", len(text))
                # tokenLists: list[list[bytes]] = [
                #     [bytes([b]) for b in word.group().encode("utf-8")] for word in regex.finditer(PAT, text)
                # ]
        else:
            text = input_path
            # print("-------------text Length:", len(text))
            # tokenLists: list[list[bytes]] = [
            #         [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(PAT, text)
            #     ]
            #将句子按照正则表达式切分成单词，再将单词转化为byte列表，这样就得到了tokenLists

        pre_tokens_cnt = defaultdict(int)
        chunks = regex.split("|".join(map(re.escape, special_tokens)), text)
        
        for chunk in chunks:
            for temp in regex.finditer(PAT, chunk):
                word = temp.group(0)
                #word是单词字符串,group()函数返回匹配的字符串，这里假设word是"hello"
                pre_tokens_cnt[word_to_bytes_tuple(word)] += 1

        ansMerges = []
        ansVocabs = {i : bytes([i]) for i in range(256)}
        # vocabs是由“字词”->id组成的字典，ansVocabs是由id->“字词”组成的字典
        # 前者用于decode，后者用于返回函数符合要求的vocab

        special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        for token_bytes in special_tokens_bytes:
            if token_bytes not in ansVocabs.values():
                ansVocabs[len(ansVocabs)] = token_bytes
        
        
        print("encode Start------------------")
        for i in range(num_merges):
            counts = get_counts(pre_tokens_cnt)
            
            if not counts:
                print(f"Early stop at {i} merges (no more pairs)")
                break

            # 牛魔的搁这搞半天，还得考虑多个pair并列的情况，然后取字典序最大的那个，怪不得其他都对的上就数据的位置对不上
            max_count = max(counts.values())
            top_pairs = [pair for pair, count in counts.items() if count == max_count]
            top_pair = max(top_pairs)
            
            new_tokenId = len(ansVocabs)
            token_bytes = top_pair[0] + top_pair[1]
            
            # print(f"{top_pair[0]} {top_pair[1]} {new_tokenId - 255}")
            ansMerges.append((top_pair[0], top_pair[1]))
            ansVocabs[new_tokenId] = token_bytes


            for old_token, new_token, cnt in merge(pre_tokens_cnt, top_pair):
                pre_tokens_cnt[new_token] += cnt
                del pre_tokens_cnt[old_token]

        return ansVocabs, ansMerges
        # print("token length", len(token))
        # print("ids length", len(ids))
        # print("compression rate", len(token) / len(ids)) #我们会发现ids经过压缩后，长度大大缩短了。这就说明原始token经过BPE算法后，部分字符成功完成了合并。
    
    return encode(input_path, vocab_size, special_tokens)
    # raise NotImplementedError
    
    





    
    
    
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import re
    import regex
    import collections as Collections
    
    def get_counts(tokenLists):
        counts = defaultdict(int)
        for tokens in tokenLists:
            for pair in zip(tokens[:-1], tokens[1:]):
                counts[pair] += 1
        return counts
    
    def merge(tokenLists, pair_target):
        newTokens = []
        
        for tokens in tokenLists:
            i = 0
            new_word = []
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair_target:
                    new_word.append(pair_target[0] + pair_target[1])
                    i += 2
                else:
                    new_word.append(tokens[i])
                    i += 1
            if i == len(tokens) - 1:
                new_word.append(tokens[i])
                
            newTokens.append(new_word)
            
        return newTokens

    # 接下来的改进路径是降低时间复杂度！
    def encode(input_path, vocab_size, special_tokens):
        num_merges = vocab_size - 256 - len(special_tokens)# vocab_size是我们期望的词汇表库的大小，256是Unicode编码上限，还需要给特殊字符留位置
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 + len(special_tokens)")
        
        # print(f"num merges: {num_merges}")
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if isinstance(input_path, os.PathLike):
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
                # print("-------------text Length:", len(text))
                tokenLists: list[list[bytes]] = [
                    [bytes([b]) for b in word.group().encode("utf-8")] for word in regex.finditer(PAT, text)
                ]
        else:
            text = input_path
            # print("-------------text Length:", len(text))
            tokenLists: list[list[bytes]] = [
                    [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(PAT, text)
                ]
            #将句子按照正则表达式切分成单词，再将单词转化为byte列表，这样就得到了tokenLists
            
    
        ansMerges = []
        ansVocabs = {i : bytes([i]) for i in range(256)}
        # vocabs是由“字词”->id组成的字典，ansVocabs是由id->“字词”组成的字典
        # 前者用于decode，后者用于返回函数符合要求的vocab

        special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        for token_bytes in special_tokens_bytes:
            if token_bytes not in ansVocabs.values():
                ansVocabs[len(ansVocabs)] = token_bytes
        
        
        print("encode Start------------------")
        for i in range(num_merges):
            counts = get_counts(tokenLists)
            
            if not counts:
                print(f"Early stop at {i} merges (no more pairs)")
                break

            # 牛魔的搁这搞半天，还得考虑多个pair并列的情况，然后取字典序最大的那个，怪不得其他都对的上就数据的位置对不上
            max_count = max(counts.values())
            top_pairs = [pair for pair, count in counts.items() if count == max_count]
            top_pair = max(top_pairs)
            
            new_tokenId = len(ansVocabs)
            token_bytes = top_pair[0] + top_pair[1]
            
            # print(f"{top_pair[0]} {top_pair[1]} {new_tokenId - 255}")
            ansMerges.append((top_pair[0], top_pair[1]))
            ansVocabs[new_tokenId] = token_bytes
            
            
            tokenLists = merge(tokenLists, top_pair) 

        return ansVocabs, ansMerges
        # print("token length", len(token))
        # print("ids length", len(ids))
        # print("compression rate", len(token) / len(ids)) #我们会发现ids经过压缩后，长度大大缩短了。这就说明原始token经过BPE算法后，部分字符成功完成了合并。
    
    return encode(input_path, vocab_size, special_tokens)
    # raise NotImplementedError