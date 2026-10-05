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
END_OF_TEXT = "<|endoftext|>"


def pretokenize(text: str) -> List[str]:
    return PRETOKEN_RE.findall(text)


def _bytes_to_unicode() -> Dict[int, str]:
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
    """Byte-level BPE: 256 byte tokens, then merges, then optional special tokens.

    Special tokens (e.g. END_OF_TEXT) take the last ids. encode() never produces them
    from text -- the literal string "<|endoftext|>" in a document is encoded as bytes --
    so a document boundary can only come from special_id(). A tokenizer without special
    tokens serialises exactly as before, so existing tokenizer hashes are unchanged.
    """

    def __init__(self, merges: Sequence[Tuple[str, str]] = (), seen_bytes: Iterable[int] = range(256),
                 special_tokens: Sequence[str] = ()) -> None:
        self.seen_bytes: List[int] = sorted(set(int(b) for b in seen_bytes))
        self._seen = set(self.seen_bytes)
        self.merges: List[Tuple[str, str]] = [(a, b) for a, b in merges]
        self.special_tokens: List[str] = list(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be distinct")
        n_regular = len(BASE_TOKENS) + len(self.merges)
        self.itos: List[str] = BASE_TOKENS + [a + b for a, b in self.merges] + self.special_tokens
        self.special_ids: Dict[str, int] = {t: n_regular + i for i, t in enumerate(self.special_tokens)}
        self.stoi: Dict[str, int] = {}
        for i, token in enumerate(self.itos[:n_regular]):
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

    @classmethod
    def from_word_counts(cls, words: Counter, seen_bytes: Iterable[int], vocab_size: int,
                         special_tokens: Sequence[str] = ()) -> "SubwordTokenizer":
        """Learn merges from counts of pre-tokenized words (see count_words), so a large
        corpus can be streamed: memory grows with the number of distinct words, not bytes."""
        n_merges = vocab_size - len(BASE_TOKENS) - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} leaves no room for 256 bytes + "
                             f"{len(special_tokens)} special token(s)")
        return cls(learn_merges(words, n_merges), seen_bytes, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def special_id(self, token: str = END_OF_TEXT) -> int:
        if token not in self.special_ids:
            raise KeyError(f"this tokenizer has no special token {token!r} "
                           f"(it has {self.special_tokens or 'none'})")
        return self.special_ids[token]

    @property
    def alphabet(self) -> List[str]:
        return [chr(b) for b in self.seen_bytes if b < 128]

    def is_known(self, token_id: int) -> bool:
        token_id = int(token_id)
        if token_id in self.special_ids.values():
            return False
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
        specials = set(self.special_ids.values())
        data = b"".join(self.token_bytes(int(i)) for i in ids if int(i) not in specials)
        return data.decode("utf-8", errors="replace")

    def to_dict(self) -> Dict[str, Any]:
        data = {"name": "subword", "kind": "byte_bpe", "itos": self.itos,
                "merges": [list(m) for m in self.merges], "seen_bytes": self.seen_bytes}
        if self.special_tokens:
            data["special_tokens"] = list(self.special_tokens)
        return data

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SubwordTokenizer":
        tokenizer = cls([tuple(m) for m in data["merges"]], data.get("seen_bytes", range(256)),
                        data.get("special_tokens", ()))
        if tokenizer.itos != list(data["itos"]):
            raise ValueError("subword tokenizer's itos does not match its merges")
        return tokenizer

    def describe(self) -> str:
        longest = sorted((self.decode([i]) for i in range(len(BASE_TOKENS), self.vocab_size)),
                         key=len, reverse=True)[:5]
        return (f"SubwordTokenizer (byte-level BPE) | vocab_size={self.vocab_size} | 256 bytes "
                f"+ {len(self.merges)} merges | longest pieces: {longest!r}")


def count_words(texts: Iterable[str]) -> Tuple[Counter, set]:
    """Pre-tokenize a stream of documents into word counts and the set of bytes seen."""
    words: Counter = Counter()
    seen = {ord("\n")}
    for text in texts:
        for word in pretokenize(text):
            words[_to_symbols(word)] += 1
        seen.update(text.encode("utf-8"))
    return words, seen


def learn_merges(words: Counter, n_merges: int) -> List[Tuple[str, str]]:
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
