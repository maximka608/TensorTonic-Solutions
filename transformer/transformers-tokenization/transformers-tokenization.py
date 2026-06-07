from typing import Dict, List


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            for word in words:
                unique_words.add(word)

        unique_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ] + sorted(unique_words)

        for idx, token in enumerate(unique_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        self.vocab_size = len(unique_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.lower().split()
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.word_to_id["<UNK>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        tokens = ""
        for i in range(len(ids)):
            if ids[i] in self.id_to_word:
                tokens += self.id_to_word[ids[i]]
            else:
                tokens += self.unk_token
            if i < len(ids) - 1:
                tokens += " "
        return tokens