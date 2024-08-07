import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from features.tokenizer import CharacterTokenizer


class BiCPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden features.
        vocab_size: Size of vocabulary
        use_embedding: Whether to use embedding layer or one-hot encoding
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate

    """
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, use_embedding: bool = False, rank: int = 8,
                 tokenizer: CharacterTokenizer = None, batch_first: bool = True, dropout: float = 0.5,
                 gate: str = 'tanh', **kwargs):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank
        self.gate = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": lambda x: x}[gate]

        # Define embedding and decoder layers
        if use_embedding:
            self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        else:
            # One hot version
            self.embedding = lambda x: torch.nn.functional.one_hot(x, vocab_size).float()
            self.input_size = vocab_size

        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.vocab_size)
        )

        # Encoder using CP factors
        self.A = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.C = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.d = nn.Parameter(torch.Tensor(self.hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        return h

    def predict(self, inp: Union[torch.LongTensor, str], init_states: tuple = None, top_k: int = 1,
                device=torch.device('cpu')):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not defined. Please provide a tokenizer to the model.")
                x = torch.tensor(self.tokenizer.char_to_ix(inp)).reshape(1, 1).to(device)
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, inp: torch.LongTensor, init_states: torch.Tensor = None):

        if self.batch_first:
            inp = inp.transpose(0, 1)

        if len(inp.shape) != 2:
            raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))

        x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        sequence_length, batch_size, _ = x.size()
        hidden_seq = []

        device = x.device

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        else:
            h_t = init_states
            h_t = h_t.to(device)

        for t in range(sequence_length):
            x_t = x[t, :, :]

            A_prime = h_t @ self.A
            B_prime = x_t @ self.B

            h_t = self.gate(
                torch.einsum("br,br,hr -> bh", A_prime, B_prime, self.C) + self.d
            )


            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        output = self.decoder(hidden_seq.contiguous())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_t
