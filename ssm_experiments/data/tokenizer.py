import string
import torch


class NumberTokenizer:
    def __init__(self, TO_TOKEN, TO_CHAR):
        self.TO_TOKEN = TO_TOKEN
        self.TO_CHAR = TO_CHAR
        self.bos_token_id = TO_TOKEN['$']
        self.eos_token_id = TO_TOKEN['.']

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def decode(self, x):
        x = x.detach().cpu().numpy()
        decoded = ''.join([str(t) if t not in self.TO_CHAR else self.TO_CHAR[t] for t in x])
        return decoded

    def __len__(self):
        return len(self.TO_TOKEN)


def get_tokenizer(args):
    string_ascii_lowercase = string.ascii_lowercase[: args.vocab_size]
    letters = dict(zip(string_ascii_lowercase, range(args.vocab_size)))
    symbols = {'$': len(letters), '|': len(letters) + 1, '.': len(letters) + 2, '*': len(letters) + 3}

    TO_TOKEN = {**letters, **symbols}
    TO_CHAR = {v: k for k, v in TO_TOKEN.items()}

    tokenizer = NumberTokenizer(TO_TOKEN, TO_CHAR)
    return tokenizer, TO_TOKEN, TO_CHAR

