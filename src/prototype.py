import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
import numpy as np

class ReceiptEmbedder:
    def __init__(self):
        # We use a pre-trained ResNet50 as our feature extractor. 
        # Transfer learning from ImageNet provides a strong baseline for detecting 
        # structural features (edges, shapes, textures) without requiring our own massive dataset.
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        
        # We strip the final classification layer (fc) because we don't want to classify 
        # the image into one of ImageNet's 1000 categories (dogs, cars, etc.). 
        # Instead, we want the raw 2048-dimensional continuous latent representation 
        # that captures the visual "fingerprint" of the receipt's layout and content.
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        
        # Move to GPU if available to drastically speed up bulk embedding generation.
        # This will be critical when processing backlogs of thousands of receipts.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # These specific normalization values are required because the ResNet model 
        # was originally trained on images normalized with these exact statistics. 
        # Deviating from them would cause a domain shift, degrading the embedding quality.
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image_path):
        try:
            # Convert to RGB to handle grayscale or CMYK images, ensuring consistent 3-channel input.
            # Discarding alpha channels (RGBA) prevents tensor shape mismatches.
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad(): # Disable gradient tracking to save memory and speed up inference
                embedding = self.model(img_tensor)
            
            # Flatten the [1, 2048, 1, 1] tensor into a 1D array of 2048 features
            vector = embedding.squeeze().cpu().numpy()
            
            # L2 normalization projects all embedding vectors onto a unit hypersphere.
            # This is a mathematical trick that makes the Inner Product of two vectors 
            # exactly equal to their Cosine Similarity. It makes similarity search scale-invariant 
            # (e.g., ignoring absolute brightness differences between two photos of the same receipt).
            faiss.normalize_L2(vector.reshape(1, -1))
            return vector
            
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            return None

class FraudIndex:
    def __init__(self, embedding_dim=2048):
        # We choose IndexFlatIP (Inner Product) instead of L2 distance because 
        # Cosine Similarity is generally more robust for high-dimensional semantic vectors.
        # Combined with our L2-normalized vectors above, this gives exact Cosine distance.
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # In this prototype, we store paths in a list where the list index matches the FAISS ID.
        # In production, this would be replaced by mapping FAISS IDs to a MongoDB document ID.
        self.image_paths = []

    def add_images(self, embedder, folder_path):
        vectors = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder_path, file)
                vec = embedder.get_embedding(full_path)
                if vec is not None:
                    vectors.append(vec)
                    self.image_paths.append(full_path)
        
        if vectors:
            # FAISS requires a contiguous float32 numpy array for C++ level optimizations
            vector_matrix = np.vstack(vectors).astype('float32')
            self.index.add(vector_matrix)
            print(f"Added {len(vectors)} images to the index. Total index size: {self.index.ntotal}")

    def find_duplicates(self, embedder, query_image_path, threshold=0.95):
        # The threshold of 0.95 is an aggressive starting heuristic for "almost identical".
        # Values below this might just be similar templates (e.g. two different Uber receipts), 
        # not actual fraud/duplicates.
        query_vec = embedder.get_embedding(query_image_path)
        if query_vec is None: return []

        query_matrix = query_vec.reshape(1, -1).astype('float32')
        
        # k=5 means we retrieve the top 5 closest matches in the vector space. 
        # We filter by threshold later so we don't accidentally flag completely dissimilar items.
        distances, indices = self.index.search(query_matrix, k=5)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist >= threshold:
                results.append((self.image_paths[idx], dist))
        return results

if __name__ == "__main__":
    print("Initializing Receipt Embedder and FAISS Index...")
    embedder = ReceiptEmbedder()
    index = FraudIndex()
    
    # ---------------------------------------------------------
    # Example usage flow for when the dataset is ready:
    # ---------------------------------------------------------
    # dataset_dir = "./dataset"
    # if os.path.exists(dataset_dir):
    #     index.add_images(embedder, dataset_dir)
    #     
    #     # Test with an image you've slightly altered (cropped/brightened)
    #     test_img = "./dataset/suspicious_upload.jpg"
    #     if os.path.exists(test_img):
    #         matches = index.find_duplicates(embedder, test_img)
    #         print(f"Matches for {test_img}:")
    #         for path, score in matches:
    #             print(f" -> {path} (Similarity: {score:.4f})")
    # else:
    #     print("Create a './dataset' folder and add images to test the prototype.")
