import torch
from torch import nn
from src import config


class TranslationEncoder(nn.Module):

    def __init__(self,vocab_size,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru=nn.GRU(input_size=config.EMBEDDING_DIM,
                        hidden_size=config.HIDDEN_SIZE,
                        batch_first=True)


    def forward(self,x):
        embedded=self.embedding(x)
        output,_=self.gru(embedded)
        lengths=(x!=self.embedding.padding_idx).sum(dim=1)
        last_hidden=output[torch.arange(output.shape[0]), lengths-1]
        return last_hidden

class TranslationDecoder(nn.Module):

    def __init__(self,vocab_size,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru=nn.GRU(input_size=config.EMBEDDING_DIM,
                        hidden_size=config.HIDDEN_SIZE,
                        batch_first=True)
        self.fc=nn.Linear(config.HIDDEN_SIZE,vocab_size)

    def forward(self,x,hidden):
        embedded=self.embedding(x)
        output,hidden_n =self.gru(embedded,hidden)
        output=self.fc(output)
        return output,hidden_n

class TranslationModel(nn.Module):

    def __init__(self,zh_vocab_size,en_vocab_size,zh_padding_idx,en_padding_idx):
        super().__init__()
        self.encoder=TranslationEncoder(zh_vocab_size,zh_padding_idx)
        self.decoder=TranslationDecoder(en_vocab_size,en_padding_idx)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, hidden):
        return self.decoder(x, hidden)