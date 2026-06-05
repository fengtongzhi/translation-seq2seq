import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持 python src/evaluate.py 和 from src 两种调用方式
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from nltk.translate.bleu_score import corpus_bleu

from src import config
from src.dataset import get_dataloaders
from src.model import TranslationModel
from src.predict import predict_batch
from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def evaluate(model, test_dataloader, device,en_tokenizer):

    predictions = []
    references = []

    for inputs, targets, zh_lens, en_lens in test_dataloader:
        inputs = inputs.to(device)
        # inputs.shape: (batch_size, seq_len)
        targets = targets.tolist()

        batch_results = predict_batch(model, inputs, en_tokenizer)
        # batch_results.shape: (batch_size, seq_len)

        predictions.extend(batch_results)
        references.extend([[target[1:target.index(en_tokenizer.eos_token_index)]] for target in targets])

    return corpus_bleu(references, predictions)

def run_evaluate():
    # 准备工作
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')

    # 模型
    model=TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index, en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR /'best_model.pth'))

    # 加载测试数据
    test_dataloader = get_dataloaders(train=False)

    # 评估逻辑
    bleu = evaluate(model, test_dataloader, device, en_tokenizer)
    print(f"bleu: {bleu:.4f}")

if __name__ == '__main__':
    run_evaluate()