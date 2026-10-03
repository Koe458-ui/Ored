from __future__ import annotations

import pytest

from ored.data.tokenizer import (UNK_TOKEN, CharTokenizer, SubwordTokenizer, Tokenizer,
                                 build_tokenizer, pretokenize)


@pytest.fixture()
def tokenizer():
    return CharTokenizer.from_text("the cat sees 13 + 8 = 21 .\n")


def test_round_trip_is_exact(tokenizer):
    for text in ["the cat", "13 + 8 = 21", "\n", "the cat sees 13 .\n"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_is_deterministic():
    a = CharTokenizer.from_text("banana split")
    b = CharTokenizer.from_text("banana split")
    assert a.itos == b.itos


def test_vocab_ordering_does_not_depend_on_text_order():
    a = CharTokenizer.from_text("abc")
    b = CharTokenizer.from_text("cba")
    assert a.itos == b.itos


def test_unknown_token_is_id_zero(tokenizer):
    assert tokenizer.itos[0] == UNK_TOKEN
    assert tokenizer.encode("Z") == [0]


def test_unknown_characters_do_not_crash(tokenizer):
    assert tokenizer.decode(tokenizer.encode("the ZZZ cat")) == "the  cat"


def test_vocab_size_counts_the_unknown_token(tokenizer):
    assert tokenizer.vocab_size == len(tokenizer.itos)
    assert tokenizer.vocab_size == len(set("the cat sees 13 + 8 = 21 .\n")) + 1


def test_out_of_range_id_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="outside this tokenizer"):
        tokenizer.decode([tokenizer.vocab_size + 5])


def test_save_and_load(tokenizer, tmp_path):
    path = tokenizer.save(tmp_path / "tok.json")
    reloaded = Tokenizer.load(path)
    assert reloaded.itos == tokenizer.itos
    assert reloaded.encode("the cat") == tokenizer.encode("the cat")


def test_dict_round_trip(tokenizer):
    rebuilt = Tokenizer.from_dict(tokenizer.to_dict())
    assert rebuilt.decode(rebuilt.encode("13 + 8")) == "13 + 8"


def test_build_tokenizer_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown tokenizer"):
        build_tokenizer("bpe_that_does_not_exist_yet", "text")


SUBWORD_TEXT = "the cat sees the cats .\nthe cat sat on 13 + 8 = 21 .\nWhat is velocity? It's fast.\n\n" * 10


@pytest.fixture()
def subword():
    return build_tokenizer("subword", SUBWORD_TEXT, vocab_size=320)


def test_subword_round_trip_is_exact_for_any_text(subword):
    for text in [SUBWORD_TEXT, "the cat", "\n", "13 + 8 = 21", "  two  spaces\t",
                 "héllo wörld", "日本語 😀", "Zebra!\r\n"]:
        assert subword.decode(subword.encode(text)) == text


def test_subword_starts_from_the_256_bytes(subword):
    assert subword.vocab_size <= 320
    assert [subword.token_bytes(i) for i in range(256)] == [bytes([b]) for b in range(256)]
    assert subword.encode("😀") == list("😀".encode("utf-8"))


def test_subword_pieces_are_longer_than_characters(subword):
    assert len(subword.encode(SUBWORD_TEXT)) < len(SUBWORD_TEXT) / 2
    assert len(subword.encode(" velocity")) == 1


def test_subword_reads_digits_one_at_a_time(subword):
    assert [subword.decode([i]) for i in subword.encode("21")] == ["2", "1"]


def test_subword_is_deterministic():
    a = build_tokenizer("subword", SUBWORD_TEXT, vocab_size=300)
    b = build_tokenizer("subword", SUBWORD_TEXT, vocab_size=300)
    assert a.to_dict() == b.to_dict()


def test_subword_needs_room_for_the_bytes():
    with pytest.raises(ValueError, match="vocab_size >= 256"):
        build_tokenizer("subword", SUBWORD_TEXT, vocab_size=100)


def test_subword_dict_round_trip(subword, tmp_path):
    reloaded = Tokenizer.load(subword.save(tmp_path / "tok.json"))
    assert isinstance(reloaded, SubwordTokenizer)
    assert reloaded.itos == subword.itos
    assert reloaded.encode(SUBWORD_TEXT) == subword.encode(SUBWORD_TEXT)


def test_pretokenize_covers_every_character():
    text = "héllo  wörld__x\t\ty! 3.14\n\n¿qué? it's\r\n"
    assert "".join(pretokenize(text)) == text
    assert "'s" in pretokenize(text)


def test_subword_knows_only_the_bytes_it_was_trained_on(subword):
    assert all(subword.is_known(i) for i in subword.encode(SUBWORD_TEXT))
    assert not any(subword.is_known(i) for i in subword.encode("日本"))
    assert Tokenizer.from_dict(subword.to_dict()).seen_bytes == subword.seen_bytes
