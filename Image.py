import pandas as pd
from IPython.display import Image as DisplayImage,display
from PIL import Image
import open_clip
import numpy as np
import torch.nn.functional as F
import torch
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# Directory to save results
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Load the precomputed embeddings DataFrame
df = pd.read_pickle('image_embeddings.pickle')

# Load the tokenizer and pre-trained model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if i % 10 != 0:
            continue
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names



# Function to search for top-k images using text query
def search_by_text(text_query, top_k=5):
    text = tokenizer([text_query])
    query_embedding = F.normalize(model.encode_text(text), dim=1)

    embeddings = torch.stack([torch.tensor(e) for e in df['embedding']])
    similarities = torch.mm(query_embedding, embeddings.T).squeeze(0)
    print(df.head())
    top_k_indices = similarities.topk(top_k).indices.cpu().detach().numpy()
    top_k_scores = similarities.topk(top_k).values.cpu().detach().numpy()
    
    return [(df.iloc[idx]['file_name'], score) for idx, score in zip(top_k_indices, top_k_scores)]

# Function to search for top-k images using combined query (image + text)
def combined_query(image_path, text_query, lam=0.8, top_k=5):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image), dim=1)

    text = tokenizer([text_query])
    text_query = F.normalize(model.encode_text(text), dim=1)

    query = F.normalize(lam * text_query + (1.0 - lam) * image_query, dim=1)

    embeddings = torch.stack([torch.tensor(e) for e in df['embedding']])
    similarities = torch.mm(query, embeddings.T).squeeze(0)

    top_k_indices = similarities.topk(top_k).indices.cpu().detach().numpy()
    top_k_scores = similarities.topk(top_k).values.cpu().detach().numpy()
    
    return [(df.iloc[idx]['file_name'], score) for idx, score in zip(top_k_indices, top_k_scores)]

def image_pca(image_path, pca, reduced_embeddings, train_image_names, top_k=5):
    image = Image.open(image_path)
    img = image.convert('L')  # Convert to grayscale ('L' mode)
    img = img.resize((224, 224))  # Resize to target size
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image_query = pca.transform(img_array.flatten().reshape(1, 50176))
    print(image_query)
    distances = np.linalg.norm(reduced_embeddings - image_query, axis=1)
    nearest_indices = np.argsort(distances)[:top_k]
    top_distances = distances[nearest_indices]
    print(reduced_embeddings[nearest_indices[0]])
    return [(train_image_names[idx], score) for idx, score in zip(nearest_indices, top_distances)]

# Function to perform experiments with both text and combined queries
def do_experiments():
    '''
    # Text Query Experiment
    text_query = "cat cuddles with dog on sofa."
    text_results = search_by_text(text_query)
    
    print("\n--- Text Query Results ---")
    for impath, score in text_results:
        print(f"Image Path: {impath}, Similarity Score: {score:.4f}")
        display(DisplayImage(filename=os.path.join('C:\\Users\\cicil\\Desktop\\New folder (2)\\static\\coco_images_resized', impath)))

    # Combined Query Experiment
    image_path = "house.jpg"
    combined_query_text = "snowy"
    lam = 0.8
    combined_results = combined_query(image_path, combined_query_text, lam)
    
    print("\n--- Combined Query Results ---")
    for impath, score in combined_results:
        print(f"Image Path: {impath}, Similarity Score: {score:.4f}")
        display(DisplayImage(filename=os.path.join('C:\\Users\\cicil\\Desktop\\New folder (2)\\static\\coco_images_resized', impath)))

    # Save results
    with open(f"{result_dir}/results_summary.txt", "w") as f:
        f.write("--- Text Query Results ---\n")
        for impath, score in text_results:
            f.write(f"Image Path: {impath}, Similarity Score: {score:.4f}\n")

        f.write("\n--- Combined Query Results ---\n")
        for impath, score in combined_results:
            f.write(f"Image Path: {impath}, Similarity Score: {score:.4f}\n")
    '''
    image_path = "house.jpg"
    results = image_pca(image_path)
    print(results)

if __name__ == "__main__":
    do_experiments()
