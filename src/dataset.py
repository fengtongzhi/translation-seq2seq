import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src import config

# <pad> 在词表中总是索引 0（由 tokenizer.BaseTokenizer.build_vocab 保证）
PAD_IDX = 0


def pad_collate(batch):
    """将 batch 内的变长序列填充至本批最大长度"""
    zh_seqs, en_seqs = zip(*batch)

    zh_padded = pad_sequence(zh_seqs, batch_first=True, padding_value=PAD_IDX)
    en_padded = pad_sequence(en_seqs, batch_first=True, padding_value=PAD_IDX)

    # 返回原始长度（可用于 loss masking / teacher forcing）
    zh_lens = torch.tensor([len(s) for s in zh_seqs], dtype=torch.long)
    en_lens = torch.tensor([len(s) for s in en_seqs], dtype=torch.long)

    return zh_padded, en_padded, zh_lens, en_lens


class TranslationDataset(Dataset):
    def __init__(self, path):
        # path 可以是 pathlib.Path 或 字符串
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)

        return input_tensor, target_tensor

def collate_fn(batch):
    input_tensors=[item[0] for item in batch]
    target_tensors=[item[1] for item in batch]
    input_padded = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_tensors, batch_first=True, padding_value=0)
    return input_padded, target_padded


def get_dataloaders(train=True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset = TranslationDataset(path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=train,
        collate_fn=pad_collate,
    )
    return dataloader
