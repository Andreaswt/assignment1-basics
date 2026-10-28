
from collections import defaultdict
from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from tests.common import FIXTURES_PATH
from multiprocessing import Pool


def pre_tokenize(input_path: str, lower: int, upper: int, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    # pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(lower)
        result = f.read(upper - lower)
        chunks = re.split("|".join([re.escape(t)
                                    for t in special_tokens]), result.decode("utf-8"))

        for chunk in chunks:
            pre_tokens_iterator = re.finditer(PAT, chunk)

            for pt in pre_tokens_iterator:
                token = pt.group()
                encoded = token.encode("utf-8")
                pre_tokens[tuple(bytes([b]) for b in encoded)] += 1

    return pre_tokens


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # pre-tokenize
    pre_tokens = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 5, b"<|endoftext|>")

        # todo: pre-tokenize in parallel
        inputs = []
        for i in range(len(boundaries) - 1):
            inputs.append(
                (input_path, boundaries[i], boundaries[i+1], special_tokens))

        with Pool(processes=5) as pool:
            results = pool.starmap(pre_tokenize, inputs)

            # add this chunks pre-tokens freqs to final dict
            for result in results:
                for token, freq in result.items():
                    pre_tokens[token] += freq

    # add bytes and special tokens as part of initialization
    for i in range(256):
        vocab[i] = bytes([i])

    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8")

    # merge
    while len(vocab) < vocab_size:
        pairs = defaultdict(int)
        for token in pre_tokens:
            for i in range(len(token) - 1):
                pairs[(token[i], token[i + 1])] += pre_tokens[token]

        # sort by frequency, then lexi
        if not pairs:
            break
        top = max(pairs, key=lambda x: (pairs[x], x))
        new_pretokens = {}
        for token in pre_tokens:
            new, i = [], 0
            while i < len(token):
                if i + 1 < len(token) and top[0] == token[i] and top[1] == token[i+1]:
                    new.append(top[0] + top[1])
                    i += 2
                else:
                    new.append(token[i])
                    i += 1
            new_pretokens[tuple(new)] = pre_tokens[token]

        merges.append(top)
        vocab[len(vocab)] = top[0] + top[1]
        pre_tokens = new_pretokens

    return (vocab, merges)


if __name__ == "__main__":
    from line_profiler import LineProfiler
    lp = LineProfiler()
    profiled = lp(train_bpe)
    profiled(FIXTURES_PATH / "tinystories_sample_5M.txt",
             1000, ["<|endoftext|>"])
    lp.print_stats()

# if __name__ == "__main__":
#     vocab, merges = train_bpe(
#         FIXTURES_PATH / "tinystories_sample_5M.txt", 10000, ["<|endoftext|>"])
#     print(len(vocab), len(merges))
