import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import config
from src.dataset import get_dataloaders
from src.tokenizer import ChineseTokenizer, EnglishTokenizer
from src.model import TranslationModel

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for inputs,targets,zh_lens,en_lens in tqdm(dataloader,desc="训练中"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        decoder_inputs=targets[:,:-1]
        decoder_targets=targets[:,1:]
    #     前向传播
    #     编码阶段
        context_vector=model.encode(inputs)

        # 解码阶段
        decoder_hidden=context_vector.unsqueeze(0)

        decoder_outputs=[]
        seq_len=decoder_inputs.shape[1]
        for t in range(seq_len):
            decoder_input_t=decoder_inputs[:,t].unsqueeze(1)
            output_t,decoder_hidden=model.decode(decoder_input_t,decoder_hidden)
            decoder_outputs.append(output_t)

        decoder_outputs=torch.cat(decoder_outputs,dim=1)
        # decoder_outputs 形状: (batch_size, seq_len, vocab_size)
        decoder_outputs=decoder_outputs.reshape(-1,decoder_outputs.shape[-1])
        # decoder_outputs 形状: (batch_size*seq_len, vocab_size)
        # decoder_targets 形状: (batch_size, seq_len) -> (batch_size*seq_len,)
        decoder_targets=decoder_targets.reshape(-1)

        loss = loss_fn(decoder_outputs,decoder_targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据
    dataloader=get_dataloaders()

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / "en_vocab.txt")

    # 模型
    model = TranslationModel(zh_tokenizer.vocab_size,en_tokenizer.vocab_size,zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    best_loss = float('inf')
    for epoch in range(config.EPOCHS):
        print(f"====================Epoch {epoch + 1}/{config.EPOCHS}========================")
        loss=train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss: {loss:.4f}")

        # 记录到 TensorBoard
        writer.add_scalar("loss", loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / "best_model.pth")
            print("保存了新的最佳模型！")

    writer.close()


if __name__ == '__main__':
    train()