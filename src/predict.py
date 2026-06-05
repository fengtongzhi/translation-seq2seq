import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持 python src/predict.py 和 from src 两种调用方式
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from src.tokenizer import ChineseTokenizer, EnglishTokenizer
from src.model import TranslationModel
from src import config

def predict_batch(model, input_tensor,en_tokenizer):
    """
    对一批中文句子进行翻译。
    :param model: 翻译模型。
    :param input_tensor: 中文输入张 量，形状为 (batch_size, seq_len)。
    :param en_tokenizer: 英文分词器。
    :return: 英文 token 索引列表。
    """
    model.eval()
    with torch.no_grad():
        # 编码输入
        context_vector = model.encoder(input_tensor)
        # context_vector 的形状为 (batch_size, hidden_size)

        # 解码
        batch_size = input_tensor.size(0)

        # 隐藏状态
        decoder_hidden = context_vector.unsqueeze(0)  # (1, batch_size, hidden_size)
        # 初始输入为 <sos> token
        decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_token_index, dtype=torch.long, device=input_tensor.device)  # (batch_size, 1)

        # 预测结果缓存
        generated=[]

        # 检测每个样本是否已经生成了 <eos> token
        is_finished = torch.full([batch_size], False, device=input_tensor.device)

        # 自回归
        for i in range(config.MAX_SEQ_LEN):
            # 解码
            output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)  # output: (batch_size, vocab_size)
            predicted_token = torch.argmax(output,dim=-1)  # (batch_size,)

            # 保存预测结果
            generated.append(predicted_token)

            # 更新输入
            decoder_input = predicted_token

            # 判断是否应该结束
            is_finished |= (predicted_token.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():
                break

        # 处理预测结果
        generated = torch.cat(generated, dim=1).tolist()  # (batch_size, seq_len)

        # 去掉eos之后的部分
        for index,sentence in enumerate(generated):
            if en_tokenizer.eos_token_index in sentence:
                eos_index = sentence.index(en_tokenizer.eos_token_index)
                generated[index] = sentence[:eos_index]

        return generated

def predict(text,model, zh_tokenizer, en_tokenizer, device):
    """
    对单条中文句子进行翻译。
    :param text: 中文输入句子。
    :param model: 翻译模型。
    :param zh_tokenizer: 中文分词器。
    :param en_tokenizer: 英文分词器。
    :param device: 设备。
    :return: 英文翻译句子。
    """
    indexes = zh_tokenizer.encode(text, add_sos_eos=False)
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)

    batch_result=predict_batch(model,input_tensor,en_tokenizer)

    return en_tokenizer.decode(batch_result[0])

def run_predict():
    """
    启动交互式翻译程序。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')

    # 模型
    model=TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index, en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR /'best_model.pth'))


    print('欢迎使用翻译系统，请输入中文句子：（输入 q 或 quit 退出）')
    while True:
        user_input = input('中文：')
        if user_input in ['q', 'quit']:
            print('谢谢使用，再见！')
            break
        if not user_input:
            print('请输入内容')
            continue

        result = predict(user_input,model, zh_tokenizer, en_tokenizer, device)
        print(f'英文：{result}')

if __name__ == '__main__':
    run_predict()
