import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class ReceiptEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ReceiptEmbeddingNet, self).__init__()
        
        # 1. The Backbone (Feature Extractor)
        # We start with ResNet50 but we could freeze the early layers if our dataset is small
        # to prevent catastrophic forgetting of basic edge/shape detection.
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. The Projection Head (Dimensionality Reduction & Feature Selection)
        # We reduce the generic 2048 ImageNet features down to a compact 128-dimensional vector.
        # WHY: 
        # - Lower dimensionality makes FAISS vector search exponentially faster and cheaper.
        # - The bottleneck forces the network to discard irrelevant ImageNet features (like colors)
        #   and only keep the structural signals necessary to differentiate receipts.
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting on our small receipt dataset
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Project to lower dimension
        x = self.projection_head(x)
        
        # L2 Normalization in the forward pass.
        # WHY: Triplet loss relies on Euclidean distance or Cosine Similarity. 
        # Normalizing forces all embeddings to exist on the surface of a unit hypersphere,
        # making distance calculations stable and directly equivalent to Cosine Similarity.
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

def train_prototype_step():
    """
    This function demonstrates the training logic using Triplet Margin Loss.
    The Data Team will expand this with proper DataLoaders that generate triplets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReceiptEmbeddingNet(embedding_dim=128).to(device)
    
    # WHY TripletMarginLoss? 
    # It takes 3 inputs: Anchor (A), Positive (P), Negative (N).
    # Loss = max(Distance(A, P) - Distance(A, N) + Margin, 0)
    # It explicitly pushes the duplicate/augmented image (P) closer to the original (A),
    # while pushing the distinct receipt (N) away by at least the margin distance.
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    
    # Mock data: B=Batch Size, C=Channels, H=Height, W=Width
    batch_size = 4
    anchors = torch.randn(batch_size, 3, 224, 224).to(device)    # Original receipts
    positives = torch.randn(batch_size, 3, 224, 224).to(device)  # Cropped/skewed versions of the anchors
    negatives = torch.randn(batch_size, 3, 224, 224).to(device)  # Completely different receipts

    optimizer.zero_grad()

    # Generate the 128-dimensional embeddings for all three sets
    emb_anchor = model(anchors)
    emb_positive = model(positives)
    emb_negative = model(negatives)

    # Calculate loss and backpropagate
    loss = criterion(emb_anchor, emb_positive, emb_negative)
    loss.backward()
    optimizer.step()

    print(f"Prototype training step completed. Triplet Loss: {loss.item():.4f}")
    print(f"Embedding shape generated: {emb_anchor.shape}") # Should be [4, 128]

if __name__ == "__main__":
    train_prototype_step()
