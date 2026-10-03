from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Type

UNK_TOKEN = "<unk>"

TOKENIZER_REGISTRY: Dict[str, Type["Tokenizer"]] = {}


def register_tokenizer(name: str) -> Callable[[Type["Tokenizer"]], Type["Tokenizer"]]:
    def decorator(cls: Type["Tokenizer"]) -> Type["Tokenizer"]:
        key = name.lower()
        if key in TOKENIZER_REGISTRY:
            raise ValueError(f"tokenizer {name!r} is already registered")
        TOKENIZER_REGISTRY[key] = cls
        cls.name = key
        return cls

    return decorator


class Tokenizer:

    name: str = "base"

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def is_known(self, token_id: int) -> bool:
        """Whether this token stands for text the tokenizer saw in training."""
        return int(token_id) != 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tokenizer":
        key = data["name"].lower()
        if key not in TOKENIZER_REGISTRY:
            raise ValueError(f"unknown tokenizer {key!r}; have {sorted(TOKENIZER_REGISTRY)}")
        return TOKENIZER_REGISTRY[key]._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Tokenizer":
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@register_tokenizer("char")
class CharTokenizer(Tokenizer):

    def __init__(self, characters: Iterable[str]) -> None:
        unique = sorted(set(characters))
        if UNK_TOKEN in unique:
            unique.remove(UNK_TOKEN)

        self.itos: List[str] = [UNK_TOKEN] + unique
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str, **_: Any) -> "CharTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        unk = 0
        return [self.stoi.get(character, unk) for character in text]

    def decode(self, ids: Sequence[int]) -> str:
        pieces = []
        for token_id in ids:
            index = int(token_id)
            if not 0 <= index < len(self.itos):
                raise ValueError(
                    f"token id {index} is outside this tokenizer's vocabulary "
                    f"(0..{len(self.itos) - 1})"
                )
            pieces.append(self.itos[index])
        return "".join(p for p in pieces if p != UNK_TOKEN)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": "char", "itos": self.itos}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "CharTokenizer":
        tokenizer = cls("")
        tokenizer.itos = list(data["itos"])
        tokenizer.stoi = {s: i for i, s in enumerate(tokenizer.itos)}
        return tokenizer

    def describe(self) -> str:
        visible = "".join(c for c in self.itos[1:] if c not in "\n\t")
        return f"CharTokenizer | vocab_size={self.vocab_size} | symbols: {visible!r} + newline"


# GPT-4 style pre-splitting, written with the standard library: English
# contractions, words with the one non-letter in front of them (usually a space),
# runs of punctuation, and runs of whitespace with newlines kept together. Merges
# never cross these boundaries. Digits stay one per token (as in Llama) so sums
# are read digit by digit; GPT-4 groups up to three.
PRETOKEN_RE = re.compile(
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\w]?[^\W\d_]+"
    r"|_+"
    r"|\d"
    r"| ?[^\s\w]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
DEFAULT_SUBWORD_VOCAB = 1024


def pretokenize(text: str) -> List[str]:
    return PRETOKEN_RE.findall(text)


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's table giving each of the 256 bytes a printable character, so byte
    tokens and merges can be stored and read as ordinary strings."""
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    table = {b: chr(b) for b in printable}
    extra = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    return table


BYTE_TO_CHAR: Dict[int, str] = _bytes_to_unicode()
CHAR_TO_BYTE: Dict[str, int] = {c: b for b, c in BYTE_TO_CHAR.items()}
BASE_TOKENS: List[str] = [BYTE_TO_CHAR[b] for b in range(256)]


def _to_symbols(word: str) -> str:
    return "".join(BYTE_TO_CHAR[b] for b in word.encode("utf-8"))


@register_tokenizer("subword")
class SubwordTokenizer(Tokenizer):
    """Byte-level byte-pair encoding, the scheme behind GPT-2, GPT-4 and Llama 3.

    Text is read as UTF-8 bytes, so the first 256 ids are the 256 bytes and any
    text in any language or with any emoji can be encoded: nothing is unknown.
    The most frequent adjacent pairs inside a pre-split piece of the training text
    are then merged into longer tokens until the vocabulary reaches ``vocab_size``.
    """

    def __init__(self, merges: Sequence[Tuple[str, str]] = (), seen_bytes: Iterable[int] = range(256)) -> None:
        self.seen_bytes: List[int] = sorted(set(int(b) for b in seen_bytes))
        self._seen = set(self.seen_bytes)
        self.merges: List[Tuple[str, str]] = [(a, b) for a, b in merges]
        self.itos: List[str] = BASE_TOKENS + [a + b for a, b in self.merges]
        self.stoi: Dict[str, int] = {}
        for i, token in enumerate(self.itos):
            self.stoi.setdefault(token, i)
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: Dict[str, List[int]] = {}

    @classmethod
    def from_text(cls, text: str, vocab_size: int = DEFAULT_SUBWORD_VOCAB, **_: Any) -> "SubwordTokenizer":
        if vocab_size < len(BASE_TOKENS):
            raise ValueError(f"a byte-level tokenizer needs vocab_size >= {len(BASE_TOKENS)}, "
                             f"got {vocab_size}")
        words = Counter(_to_symbols(w) for w in pretokenize(text))
        seen = set(text.encode("utf-8")) | {ord("\n")}
        return cls(learn_merges(words, vocab_size - len(BASE_TOKENS)), seen)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def alphabet(self) -> List[str]:
        return [chr(b) for b in self.seen_bytes if b < 128]

    def is_known(self, token_id: int) -> bool:
        token_id = int(token_id)
        return token_id >= len(BASE_TOKENS) or token_id in self._seen

    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        pieces = list(_to_symbols(word))
        while len(pieces) > 1:
            best = None
            for i in range(len(pieces) - 1):
                rank = self.ranks.get((pieces[i], pieces[i + 1]))
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, i)
            if best is None:
                break
            i = best[1]
            pieces[i:i + 2] = [pieces[i] + pieces[i + 1]]
        ids = [self.stoi[p] for p in pieces]
        if len(self._cache) < 100_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in pretokenize(text):
            ids.extend(self._encode_word(word))
        return ids

    def token_bytes(self, index: int) -> bytes:
        if not 0 <= index < len(self.itos):
            raise ValueError(
                f"token id {index} is outside this tokenizer's vocabulary "
                f"(0..{len(self.itos) - 1})"
            )
        return bytes(CHAR_TO_BYTE[c] for c in self.itos[index])

    def decode(self, ids: Sequence[int]) -> str:
        data = b"".join(self.token_bytes(int(i)) for i in ids)
        return data.decode("utf-8", errors="replace")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": "subword", "kind": "byte_bpe", "itos": self.itos,
                "merges": [list(m) for m in self.merges], "seen_bytes": self.seen_bytes}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SubwordTokenizer":
        tokenizer = cls([tuple(m) for m in data["merges"]], data.get("seen_bytes", range(256)))
        if tokenizer.itos != list(data["itos"]):
            raise ValueError("subword tokenizer's itos does not match its merges")
        return tokenizer

    def describe(self) -> str:
        longest = sorted((self.decode([i]) for i in range(len(BASE_TOKENS), self.vocab_size)),
                         key=len, reverse=True)[:5]
        return (f"SubwordTokenizer (byte-level BPE) | vocab_size={self.vocab_size} | 256 bytes "
                f"+ {len(self.merges)} merges | longest pieces: {longest!r}")


def learn_merges(words: Counter, n_merges: int) -> List[Tuple[str, str]]:
    """Byte-pair encoding: repeatedly merge the most frequent adjacent pair.

    Ties go to the pair that sorts first, so the same text always gives the same
    merges in the same order.
    """
    vocab: List[List[str]] = [list(w) for w in sorted(words)]
    counts: List[int] = [words[w] for w in sorted(words)]
    pairs: Counter = Counter()
    where: Dict[Tuple[str, str], set] = defaultdict(set)
    for i, pieces in enumerate(vocab):
        for pair in zip(pieces, pieces[1:]):
            pairs[pair] += counts[i]
            where[pair].add(i)

    merges: List[Tuple[str, str]] = []
    while len(merges) < n_merges and pairs:
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        if best[1] < 2:
            break
        a, b = pair = best[0]
        merges.append(pair)
        for i in sorted(where.pop(pair, ())):
            pieces, n = vocab[i], counts[i]
            for old in zip(pieces, pieces[1:]):
                pairs[old] -= n
                if pairs[old] <= 0:
                    del pairs[old]
            merged: List[str] = []
            j = 0
            while j < len(pieces):
                if j + 1 < len(pieces) and pieces[j] == a and pieces[j + 1] == b:
                    merged.append(a + b)
                    j += 2
                else:
                    merged.append(pieces[j])
                    j += 1
            vocab[i] = merged
            for new in zip(merged, merged[1:]):
                pairs[new] += n
                where[new].add(i)
        pairs.pop(pair, None)
    return merges


def build_tokenizer(name: str, text: str, **options: Any) -> Tokenizer:
    key = name.lower()
    if key not in TOKENIZER_REGISTRY:
        raise ValueError(f"unknown tokenizer {name!r}; have {sorted(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[key].from_text(text, **options)
