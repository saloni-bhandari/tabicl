import torch
import torch.nn as nn

from layers import OneHotAndLinear

class ColEmbedding(nn.Module):
    def __init__(self, embedding_dim, nhead):
        super(ColEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.nhead = nhead

        # Embedding for each cell value in the column, we treat each cell as a scalar and embed it to a vector of size embedding_dim
        self.embed_cells_to_dim = nn.Linear(1, embedding_dim)

        # Transformer encoder for column-wise encoding
        # output will be the "distribution description" of the column, which captures the relationships between the cell values in the column
        # TODO: convert to SetTransformer
        self.encode_column = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead) 

        # take the distribution description and generate W and B
        ## W is the embedding_dim since we want to do element-wise multiplication with the original cell embedding
        ## B is also embedding_dim since we want to do element-wise addition with the original cell embedding
        self.generate_W = nn.Linear(embedding_dim, embedding_dim)
        self.generate_B = nn.Linear(embedding_dim, embedding_dim)

        # combine original embedding + W and B to get distribution aware embedding for each cell
        self.final_embedding = lambda cell_embedding, W, B: cell_embedding * W + B


    def forward(self, X, y, train_size=1):
        
        # X shape: (batch_size, num_cols, num_rows) = (B, M, N)
        # num_cols is the number of features in the input data
        # num_rows is number of support rows for ICL + query row
        # batch_size is the number of examples in the batch (with ICL, each example is a table with M columns and N rows)

        batch_size, num_cols, num_rows = X.size()
        
        print(f"Original X shape: {X.shape}")        
        print(f"Input y shape: {y.shape}")

        # embed y labels to be same dimension as inputs X
        y = OneHotAndLinear(num_classes=10, embed_dim=self.embedding_dim)(y.long().view(-1, 1)).view(batch_size, num_rows, -1)
        print(f"Embedded y shape: {y.shape}")

        # Step 1: Embed each cell value to a vector of size embedding_dim
        # reshape X to (B * M * N, 1) to feed into the linear layer
        X = X.view(-1, 1)
        print(f"Reshaped X shape for embedding: {X.shape}")

        X = self.embed_cells_to_dim(X)  # shape: (B * M * N, embedding_dim)
        print(f"Cell embeddings shape: {X.shape}")

        # add label embedding to row feature embedding for only the first (num_rows - 1) rows
        ## the last row is going to be unlabelled for prediction
        y[:, train_size:, :] = 0 # zero out the label embedding for the non training rows since it's unlabelled

        # reshape X for summing with y
        X = X.view(batch_size, num_cols, num_rows, self.embedding_dim)
        y = torch.unsqueeze(y, dim=1)
        print(f"Reshaped X shape for summing with y: {X.shape}")
        print(f"Unsqueezed y shape: {y.shape}")
        X = X + y

        # Use transformer to encode the column-wise relationships
        ## We want attention to be done across each column of a table (batch), so that we can learn the relationships between the cell values in the same column.
        ## This means we have B * M sequneces (one for each column in a batch), and each sequence has N tokens (one for each cell/row in the column), and each token is represented by a vector of size embedding_dim.
        ## Reshape the embeddings in to (B * M, N, embedding_dim) to feed into the transformer encoder
        X = X.view(batch_size * num_cols, num_rows, -1)
        print(f"Reshaped X for transformer: {X.shape}")

        ## Supposed to be set-transformer but we just use normal one for now
        ## Get distribution aware embedding for each column by using transformer
        X = self.encode_column(X)
        print(f"Encoded column representations shape: {X.shape}")

        # Generate W and B distribution parameters for each column
        W = self.generate_W(X)  # shape: (B * M, N, embedding_dim)
        B = self.generate_B(X)  # shape: (B * M, N, embedding_dim)

        # Combine original cell embedding with W and B to get distribution aware embedding for each cell
        ## Reshape W and B back to (B, M, N, embedding_dim)
        X = self.final_embedding(X, W, B)
        print(f"Final distribution-aware embedding shape: {X.shape}")

        # split the embeddings back into their individual batches
        X = X.view(batch_size, num_cols, num_rows, -1) # (B, M, N, embedding_dim)
        print(f"Final output shape: {X.shape}")
        
        return X

if __name__ == "__main__":
    # testing
    batch_size = 2
    num_cols = 3
    num_rows = 4
    embedding_dim = 8
    nhead = 2

    model = ColEmbedding(embedding_dim, nhead)
    X = torch.randn(batch_size, num_cols, num_rows)
    y = torch.randint(0, 10, (batch_size, num_rows)).to(dtype=torch.float32)
    output = model(X, y)
