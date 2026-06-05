import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.tokenizer import ChineseTokenizer, EnglishTokenizer



def process():
    print("开始处理数据...")
    # 读取文件
    # 原始数据格式(3列, Tab分隔): 列0=英文, 列1=中文, 列2=版权信息
    # 只取列0(英文)和列1(中文), 分别命名为'en'和'zh'
    df = pd.read_csv(config.RAW_DATA_DIR / "cmn.txt", encoding="utf-8", usecols=[0, 1],
                     names=['en', 'zh'], sep='\t', header=None).dropna()

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 构建词表: 用训练集的英文文本构建英文词表, 用训练集的中文文本构建中文词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.MODELS_DIR / "zh_vocab.txt")
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.MODELS_DIR / "en_vocab.txt")

    # 构建 Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / "en_vocab.txt")

    # 构建训练集
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 保存训练集
    train_df.to_json(config.PROCESSED_DATA_DIR / "train.jsonl", orient='records', lines=True, force_ascii=False)

    # 构建测试集
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))
    # 保存测试集
    test_df.to_json(config.PROCESSED_DATA_DIR / "test.jsonl", orient='records', lines=True, force_ascii=False)

    print("数据处理完成!")
    print(f"  训练集大小: {len(train_df)}")
    print(f"  测试集大小: {len(test_df)}")
    print(f"  中文词表大小: {zh_tokenizer.vocab_size}")
    print(f"  英文词表大小: {en_tokenizer.vocab_size}")


if __name__ == '__main__':
    process()