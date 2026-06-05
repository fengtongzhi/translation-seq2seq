
from tqdm import tqdm

class BaseTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)
        self.word2index = {words: index for index, words in enumerate(vocab_list)}
        self.index2word = {index: words for index, words in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def tokenize(cls, text) -> list:
        raise NotImplementedError

    def encode(self, text, add_sos_eos=False):
        tokens = self.tokenize(text)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        indexes = [self.word2index.get(token, self.unk_token_index) for token in tokens]
        return indexes

    def decode(self, indexes, skip_special=False):
        """将索引序列解码为字符串
        skip_special: 是否跳过 <pad>, <sos>, <eos>, <unk> 等特殊标记
        """
        special_tokens = {self.pad_token, self.unk_token, self.sos_token, self.eos_token}
        tokens = [self.index2word.get(index, self.unk_token) for index in indexes]
        if skip_special:
            tokens = [t for t in tokens if t not in special_tokens]
        return ' '.join(tokens)

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = [cls.pad_token, cls.unk_token, cls.eos_token, cls.sos_token] + sorted(
            [token for token in vocab_set if token.strip() != '']
        )

        # 保存词表，便于后面再次使用
        with open(vocab_path, "w", encoding='utf-8') as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, "r", encoding='utf-8') as f:
            vocab_list = f.read().splitlines()
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text):
        """中文按字符级分词"""
        return list(text)


class EnglishTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text):
        """英文按单词级分词，优先使用 NLTK word_tokenize，失败时降级为 split"""
        try:
            import nltk
            return nltk.word_tokenize(text.lower())
        except (ImportError, LookupError):
            # NLTK 未安装或 punkt 数据未下载，使用简单空格分词
            return text.lower().split()