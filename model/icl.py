from typing import Optional
import torch
import torch.nn as nn

from .encoders import Encoder


class ICLearning(nn.Module):

    def __init__(
        self,
        embedding_dim=32,
        nhead=1,
        num_blocks=12,
        dim_feedforward=2048,
        vocab_size=100,
        debug=False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.debug = debug

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.transformer = Encoder(
            num_blocks=num_blocks,
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            use_rope=True,
            rope_base=100000,
            rope_interleaved=True,
            recompute=False,
        )

        self.prediction_MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, vocab_size),
        )

    def _debug_print(self, *args):
        if self.debug:
            print(*args)

    def _build_attention_mask(self, T, train_size, device):
        """
        Implements the ICL masking rule:

        train rows -> only attend to train rows
        test rows -> attend to train rows + itself
        test rows cannot attend to other test rows
        """

        mask = torch.zeros(T, T, device=device)

        # training rows cannot see test rows
        mask[:train_size, train_size:] = float("-inf")

        # test rows cannot see other test rows
        mask[train_size:, train_size:] = float("-inf")

        # but allow self attention
        for i in range(train_size, T):
            mask[i, i] = 0

        return mask

    def forward(
        self,
        R: torch.Tensor,
        y_train: torch.Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config=None,
    ):
        """
        Parameters
        ----------
        R : (B, T, D)
            row representations from RowInteraction

        y_train : (B, train_size)

        Returns
        -------
        predictions : (B, test_size, vocab_size)
        """

        B, T, D = R.shape
        train_size = y_train.shape[1]

        self._debug_print("Row representations:", R.shape)
        self._debug_print("y_train:", y_train.shape)

        # reconstruct full label tensor
        y_full = torch.zeros(B, T, dtype=torch.long, device=R.device)
        y_full[:, :train_size] = y_train

        y_emb = self.embedding(y_full)

        # hide test labels
        y_emb[:, train_size:, :] = 0

        self._debug_print("Label embeddings:", y_emb.shape)

        # combine row representation with label embedding
        rep = R + y_emb

        self._debug_print("Combined representations:", rep.shape)

        # build ICL mask
        mask = self._build_attention_mask(T, train_size, R.device)

        self._debug_print("Mask:", mask)

        # transformer
        for block in self.transformer.blocks:
            rep = block(rep, attn_mask=mask)

        self._debug_print("After transformer:", rep.shape)

        # predictions only for test rows
        test_reps = rep[:, train_size:, :]

        logits = self.prediction_MLP(test_reps)

        self._debug_print("Logits:", logits.shape)

        if not return_logits:
            logits = torch.softmax(logits / softmax_temperature, dim=-1)

        return logits

if __name__ == "__main__":
    batch_size = 2
    num_rows = 4
    embedding_dim = 32
    train_size = 2
    vocab_size = 100

    cls_outputs = torch.randn(batch_size, num_rows, embedding_dim)
    y = torch.randint(0, vocab_size, (batch_size, num_rows))
    model = ICLearning(embedding_dim=embedding_dim, vocab_size=vocab_size, debug=True)

    outputs = model(cls_outputs, y, train_size=train_size)