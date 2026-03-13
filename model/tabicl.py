import torch
import torch.nn as nn

from .column_embedding import ColEmbedding
from .row_interaction import RowInteraction
from .icl import ICLearning


class TabICL(nn.Module):

    def __init__(
        self,
        max_classes=10,
        embed_dim=128,
        col_num_blocks=3,
        col_nhead=8,
        col_num_inds=128,
        row_num_blocks=3,
        row_nhead=8,
        row_num_cls=4,
        row_rope_base=100000,
        icl_num_blocks=12,
        icl_nhead=8,
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        debug=False,
        **kwargs
    ):
        super().__init__()

        self.debug = debug
        self.embed_dim = embed_dim
        self.max_classes = max_classes

        self.col_embedder = ColEmbedding(
            embedding_dim=embed_dim,
            nhead=col_nhead,
            num_classes=max_classes,
            num_blocks=col_num_blocks,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            debug=debug,
        )

        self.row_interactor = RowInteraction(
            num_attention_blocks=row_num_blocks,
            embedding_dim=embed_dim,
            nhead=row_nhead,
            num_cls=row_num_cls,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            rope_base=row_rope_base,
            debug=debug,
        )

        self.icl_predictor = ICLearning(
            embedding_dim=embed_dim * row_num_cls,
            nhead=icl_nhead,
            num_blocks=icl_num_blocks,
            dim_feedforward=embed_dim * row_num_cls * ff_factor,
            vocab_size=max_classes,
            dropout=dropout,
            activation=activation,
            rope_base=row_rope_base,
            debug=debug,
        )

    def forward(
        self,
        X,
        y_train,
        d=None,
        embed_with_test=False,
        feature_shuffles=None,
        return_logits=True,
        softmax_temperature=0.9,
        inference_config=None,
    ):

        col_embeddings = self.col_embedder(
            X,
            y_train=y_train,
            d=d,
            embed_with_test=embed_with_test,
        )

        row_repr = self.row_interactor(
            col_embeddings,
            d=d,
        )

        logits = self.icl_predictor(
            row_repr,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
        )

        return logits


if __name__ == "__main__":

    batch_size = 2
    num_rows = 4
    num_cols = 3
    embedding_dim = 32
    train_size = 2
    vocab_size = 10

    X = torch.randn(batch_size, num_rows, num_cols)

    y_train = torch.randint(0, vocab_size, (batch_size, train_size))

    model = TabICL(vocab_size=vocab_size, embedding_dim=embedding_dim, debug=True)

    output = model(X, y_train)

    print("Output:", output.shape)
