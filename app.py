from flask import Flask, render_template, request, jsonify
import os
from Image import search_by_text, combined_query, image_pca, load_images  # Your search functions
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# Initialize Flask app
app = Flask(__name__)

image_dir = "static/coco_images_resized/"
train_images, train_image_names = load_images(image_dir, target_size=(224, 224))
pca = PCA(n_components=50)
pca.fit(train_images)
reduced_embeddings = pca.transform(train_images)

# Directories
STATIC_RESULTS_DIR = "static/results"  # Directory for static result images
os.makedirs(STATIC_RESULTS_DIR, exist_ok=True)  # Create results directory if not exists

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []  # To hold the search results
    error = None  # To hold any error messages

    if request.method == 'POST':
        try:
            # Get form data
            query_type = request.form.get('query_type')  # "Text query", "Image query", or "Hybrid query"
            text_query = request.form.get('text_query')  # Text input (optional)
            lam = float(request.form.get('lam', 0.8))   # Default hybrid weight
            top_k = int(request.form.get('top_k', 5))   # Number of results to return
            use_pca = request.form.get('image_query_pca')

            # Handle uploaded image
            image_path = None
            if 'image_query' in request.files:
                image_file = request.files['image_query']
                if image_file.filename != '':
                    filename = secure_filename(image_file.filename)
                    image_path = os.path.join(STATIC_RESULTS_DIR, filename)
                    image_file.save(image_path)

            # Perform the search based on query type
            if query_type == "Text query" and text_query:
                results = search_by_text(text_query, top_k)
            elif query_type == "Image query" and image_path:
                if use_pca == "Yes":
                    print("using image pca")
                    results = image_pca(image_path, pca, reduced_embeddings, train_image_names, top_k=top_k)  # Empty string for text query
                else:
                    results = combined_query(image_path, text_query="", lam=0.0, top_k=top_k)  # Empty string for text query
            elif query_type == "Hybrid query" and image_path and text_query:
                results = combined_query(image_path, text_query, lam=lam, top_k=top_k)
            else:
                error = "Invalid input. Please ensure all fields are filled correctly."
            print(results)
            # Clean up uploaded image (if necessary)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

        except Exception as e:
            error = str(e)

    # Render the template with results and errors
    return render_template('index.html', results=results, error=error)

if __name__ == '__main__':
    app.run(debug=True)
