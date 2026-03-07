import torch
import torch.nn as nn

class ICLearning(nn.Module):
    def __init__(self, embedding_dim=32, nhead=1, vocab_size=100):
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
        
        
    def _debug_print(self, *args):
        if self.debug:
            print(*args)

    def forward(self, cls_outputs, y, test_size):
        
        batch_size, num_rows = y.size()
        self._debug_print(f"Y shape: {y.shape}")
        
        # embed y into dimensionality of cls_outputs
        y = self.embedding(y)
        self._debug_print(f"Embedded Y shape: {y.shape}")

        # mask the last test_size labels
        y[:, -test_size: , :] = 0

        # get representation that includes label (training samples + test samples)

        self._debug_print(cls_outputs.shape, y.shape)
        rep = cls_outputs + y 
        self._debug_print(f"Representations of each row of data: {rep.shape}") # 2 batches, 4 rows each, 32 features in total (from the CLS tokens after row interaction + label embedding)
        # rep.shape=(2, 4, 32) => can consider that we have 2 sequences, of 4 tokens each, each token has 32 dim embedding

        # create mask to hide the test tokens from the train tokens, and each test token from each other
        # e.g. if 4 rows and 2 of them are test samples, row 1 attends to 1 and 2, row 2 attends to 1 and 2, 
        # row 3 attends to 1 and 2 and 3, row 4 attends to 1 and 2 and 4
        mask = torch.zeros(num_rows, num_rows)
        
        for i in range(num_rows - test_size):
            mask[i, -test_size: ] = float('-inf')
        
        mask[-test_size: , -test_size: ] = float('-inf')
        
        for i in range(test_size):
            mask[-test_size+i, -test_size+i] = 0.0
        
        self._debug_print(f"Mask: {mask}")
        self._debug_print(f"Mask shape: {mask.shape}")

        # run the rep through transformers
        rep = self.transformer(rep, mask=mask)

        self._debug_print(f"Representations after transformer {rep.shape}")

        test_reps = rep[:, -test_size:, :]

        self._debug_print(f"Test representations: {test_reps.shape}")

        # make prediction for the test tokens using their representations
        logits = self.prediction_MLP(test_reps)
        
        self._debug_print(f"Logits: {logits.shape}")

        return logits

if __name__ == "__main__":
    batch_size = 2
    num_rows = 4
    embedding_dim = 32
    test_size = 1
    vocab_size = 100

    cls_outputs = torch.randn(batch_size, num_rows, embedding_dim)
    y = torch.randint(0, vocab_size, (batch_size, num_rows))
    model = ICLearning(embedding_dim=embedding_dim, vocab_size=vocab_size)

    outputs = model(cls_outputs, y, test_size)