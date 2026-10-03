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


# Words keep the single space in front of them, digits stay one per token so sums
# are read digit by digit, and a newline is always a token of its own.
PRETOKEN_RE = re.compile(
    r"\n|[^\S\n]?[^\W\d_]+|[^\S\n]?\d|[^\S\n]?[^\s\w]+|[^\S\n]?_+|[^\S\n]+"
)
DEFAULT_SUBWORD_VOCAB = 1024


def pretokenize(text: str) -> List[str]:
    return PRETOKEN_RE.findall(text)


@register_tokenizer("subword")
class SubwordTokenizer(Tokenizer):
    """Byte-pair encoding over characters.

    Every character of the training text is a token, so nothing it has seen is
    unknown, and the most frequent adjacent pairs inside a word are merged into
    longer pieces until the vocabulary reaches ``vocab_size``.
    """

    def __init__(self, characters: Iterable[str], merges: Sequence[Tuple[str, str]] = ()) -> None:
        alphabet = sorted(set(characters))
        if UNK_TOKEN in alphabet:
            alphabet.remove(UNK_TOKEN)
        self.alphabet: List[str] = alphabet
        self.merges: List[Tuple[str, str]] = [(a, b) for a, b in merges]
        self.itos: List[str] = [UNK_TOKEN] + alphabet + [a + b for a, b in self.merges]
        self._index()

    def _index(self) -> None:
        self.stoi: Dict[str, int] = {}
        for i, s in enumerate(self.itos):
            self.stoi.setdefault(s, i)
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: Dict[str, List[int]] = {}

    @classmethod
    def from_text(cls, text: str, vocab_size: int = DEFAULT_SUBWORD_VOCAB, **_: Any) -> "SubwordTokenizer":
        characters = set(text) | {"\n"}
        characters.discard(UNK_TOKEN)
        n_merges = vocab_size - 1 - len(characters)
        words = Counter(w for w in pretokenize(text) if w != "\n")
        return cls(characters, learn_merges(words, n_merges))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        pieces = list(word)
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
        ids = [self.stoi.get(p, 0) for p in pieces]
        if len(self._cache) < 100_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in pretokenize(text):
            ids.extend(self._encode_word(word))
        return ids

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
        return {"name": "subword", "itos": self.itos,
                "alphabet": self.alphabet, "merges": [list(m) for m in self.merges]}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SubwordTokenizer":
        tokenizer = cls(data["alphabet"], [tuple(m) for m in data["merges"]])
        if tokenizer.itos != list(data["itos"]):
            raise ValueError("subword tokenizer's itos does not match its alphabet and merges")
        return tokenizer

    def describe(self) -> str:
        longest = sorted(self.itos[1 + len(self.alphabet):], key=len, reverse=True)[:5]
        return (f"SubwordTokenizer | vocab_size={self.vocab_size} | {len(self.alphabet)} characters "
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
