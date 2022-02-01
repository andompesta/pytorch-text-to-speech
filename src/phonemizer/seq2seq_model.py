import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(
        self,
        in_embeddng_size=29,
        hidden_size=256,
        out_embedding_size=74,
    ):
        super(Seq2Seq, self).__init__()
        self.in_embeddng_size = in_embeddng_size
        self.hidden_size = hidden_size
        self.out_embedding_size = out_embedding_size
        self.max_decoding_steps = 20
        self.PAD_token = 0
        self.UNK_token = 1
        self.EOS_token = 3
        self.BOS_token = 2

        self.enc_emb = nn.Embedding(
            self.in_embeddng_size, self.hidden_size, padding_idx=self.PAD_token
        )

        self.enc_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.dec_emb = nn.Embedding(
            self.out_embedding_size,
            self.hidden_size,
            padding_idx=self.PAD_token,
        )

        self.dec_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.out_embedding_size,
            bias=True,
        )

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        assert enc_input.size(0) == 1, "decoding allowed with 1 sindle word at time"
        num_batch = 1

        enc_h = torch.zeros((1, num_batch, self.hidden_size))

        enc = self.enc_emb(enc_input)
        enc_output, enc_h = self.enc_gru(enc, enc_h)

        dec_input = torch.tensor([self.BOS_token]).unsqueeze(0).long()
        dec_h = enc_h
        preds = torch.zeros(self.max_decoding_steps).long()

        for i in range(self.max_decoding_steps):
            dec = self.dec_emb(dec_input)
            dec_output, dec_h = self.dec_gru(dec, dec_h)

            logit = self.fc(dec_output)
            topv, topi = logit.data.topk(1, dim=-1)
            if topi.item() == self.EOS_token:
                break
            else:
                preds[i] = topi.item()
                dec_input = topi.detach().squeeze(0)

        return preds
