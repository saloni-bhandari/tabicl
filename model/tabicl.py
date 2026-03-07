# combine all the components into a full pipeline

import torch
import torch.nn as nn

from column_embedding import ColEmbedding
from row_embedding import RowEmbedding
from icl import ICLearning

class TabICL(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TabICL, self).__init__()
        self.column_embedding = ColEmbedding(embedding_dim=embedding_dim, nhead=2)
        self.row_embedding = RowEmbedding(num_rows=4, num_attention_blocks=4, embedding_dim=embedding_dim, nhead=2)
        self.icl = ICLearning(embedding_dim=embedding_dim*4, nhead=1, vocab_size=vocab_size) # embedding_dim*4 since we concatenate the 4 CLS tokens together for each row, and each CLS token has embedding_dim dimensions

    def forward(self, X, y, test_size): 
        # input data should be of form (batch_size, num_cols, num_rows), with labels 
        embeddings = self.column_embedding(X, y)
        print("----------")
        cls_representations = self.row_embedding(embeddings)
        print("----------")
        logits = self.icl(cls_representations, y, test_size)

        return logits

if __name__ == "__main__":
    
    batch_size = 2
    num_cols = 3
    num_rows = 4
    embedding_dim = 32

    X = torch.randn(batch_size, num_cols, num_rows)
    y = torch.randint(0, 10, (batch_size, num_rows))

    tabicl = TabICL(vocab_size=100, embedding_dim=embedding_dim)
    output = tabicl(X, y, test_size=1)
    