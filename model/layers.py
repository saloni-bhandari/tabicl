import torch
from torch import Tensor, nn
from torch.nn import functional as F

class ClassNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.classes_ = None
        self.child_nodes = []
        self.class_mapping = {}
        self.group_indices = None
        # representations
        self.R = None
        # labels
        self.y = None

    def __repr__(self):
        return f"ClassNode(depth={self.depth}, leaf={self.is_leaf})"
    
class OneHotAndLinear(nn.Linear):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(in_features=num_classes, out_features=embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, input:Tensor):
        one_hot = F.one_hot(input.long(), self.num_classes).float()
        layer = F.linear(one_hot, self.weight, self.bias)
        return layer
    
class SkippableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, skip_value = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        is_skipped = (input == self.skip_value).all(dim=-1, keepdim=True)
        return torch.where(is_skipped, self.skip_value, output).to(output.dtype)