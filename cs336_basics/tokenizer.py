from collections.abc import Iterable, Iterator
import pickle
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.inverse_vocab = {
            token: token_id for token_id, token in vocab.items()}
        self.merges = merges
        # if special_tokens contains overlaps, e.g. both <|endoftext|><|endoftext|> and <|endoftext|>
        # it should match the longest first. sorting fixes this
        self.special_tokens = sorted(
            special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        1. pretokenize: split text into tokens of bytes.
        2. run merge ops in the same order as during training
        3. lookup each merged token in the trained vocab, to find the its int ID
        """
        # pre-tokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = list()

        if self.special_tokens:
            chunks = re.split("(" + "|".join([re.escape(t)
                                              for t in self.special_tokens]) + ")", text)
        else:
            chunks = [text]

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                pre_tokens.append((chunk.encode("utf-8"),))
                continue

            pre_tokens_iterator = re.finditer(PAT, chunk)

            for pt in pre_tokens_iterator:
                token = pt.group()
                encoded = token.encode("utf-8")
                pre_tokens.append(tuple(bytes([b]) for b in encoded))

        # apply merges
        merged_tokens = list()
        # [(b't', b'h', b'e'), (b' ', b'c', b'a', b't'),]
        for token in pre_tokens:
            new_token = token
            for merge in self.merges:  # [(b't', b'h'), (b' ', b'c'),]
                # if a specific merge match a pair from token, merge
                new, i = [], 0
                # (b't', b'h', b'e') -> 1: (b't', b'h'), 2: (b'h', b'e')
                while i < len(new_token):
                    if i + 1 < len(new_token) and merge[0] == new_token[i] and merge[1] == new_token[i+1]:
                        new.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new.append(new_token[i])
                        i += 1
                new_token = tuple(new)
            merged_tokens.append(new_token)

        # inverse vocab lookup
        int_seq = list()
        for token in merged_tokens:
            for piece in token:
                int_seq.append(self.inverse_vocab[piece])

        return int_seq

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        joins together all bytes before decoding.
        utf-8 char can consist of up to 4 bytes, so decoding bytes one at a time can corrupt the decoding.
        """
        token_bytes = [self.vocab[id] for id in ids]
        return b"".join(token_bytes).decode(errors="replace")
