import torch
import torch.nn as nn

class ICLearning(nn.Module):
    def __init__(self, embedding_dim=32, test_size=1, nhead=1, vocab_size=100):
        super(ICLearning, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True),
            num_layers=12
        )
        
        self.prediction_MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, vocab_size)
        )
        self.softmax = nn.Softmax(dim=-1) # softmax across the vocab
        
        self.test_size = test_size

    def forward(self, cls_outputs, y):
        
        batch_size, num_rows = y.size()
        print(f"Y shape: {y.shape}")
        
        # embed y into dimensionality of cls_outputs
        y = self.embedding(y)
        print(f"Embedded Y shape: {y.shape}")

        # mask the last self.test_size labels
        y[:, -self.test_size: , :] = 0

        # get representation that includes label (training samples + test samples)
        rep = cls_outputs + y 
        print(f"Representations of each row of data: {rep.shape}") # 2 batches, 4 rows each, 32 features in total (from the CLS tokens after row interaction + label embedding)
        # rep.shape=(2, 4, 32) => can consider that we have 2 sequences, of 4 tokens each, each token has 32 dim embedding

        # create mask to hide the test tokens from the train tokens, and each test token from each other
        # e.g. if 4 rows and 2 of them are test samples, row 1 attends to 1 and 2, row 2 attends to 1 and 2, 
        # row 3 attends to 1 and 2 and 3, row 4 attends to 1 and 2 and 4
        mask = torch.zeros(num_rows, num_rows)
        

        for i in range(num_rows - self.test_size):
            mask[i, -self.test_size: ] = float('-inf')
        
        mask[-self.test_size: , -self.test_size: ] = float('-inf')
        
        for i in range(self.test_size):
            mask[-self.test_size+i, -self.test_size+i] = 0.0
        
        print(f"Mask: {mask}")
        print(f"Mask shape: {mask.shape}")

        # run the rep through transformers
        rep = self.transformer(rep, mask=mask)

        print(f"Representations after transformer {rep.shape}")

        test_reps = rep[:, -self.test_size:, :]

        print(f"Test representations: {test_reps.shape}")

        # make prediction for the test tokens using their representations
        logits = self.prediction_MLP(test_reps)
        
        print(f"Logits: {logits.shape}")

        return logits

if __name__ == "__main__":
    batch_size = 2
    num_rows = 4
    embedding_dim = 32
    test_size = 1
    vocab_size = 100

    cls_outputs = torch.randn(batch_size, num_rows, embedding_dim)
    y = torch.randint(0, vocab_size, (batch_size, num_rows))
    model = ICLearning(embedding_dim=embedding_dim, test_size=test_size, vocab_size=vocab_size)

    outputs = model(cls_outputs, y)