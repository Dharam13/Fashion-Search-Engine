import pandas as pd
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
import json
import os
from pathlib import Path
import weaviate
from weaviate.classes.config import Configure
from transformers import CLIPProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")


class FashionSearchEngine:
    def __init__(self, weaviate_url, weaviate_api_key):
        """
        Initialize the Fashion Search Engine

        Args:
            weaviate_url: Your Weaviate cloud cluster URL
            weaviate_api_key: Your Weaviate API key
        """
        # Initialize CLIP model
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)

        # Initialize Weaviate client
        print("Connecting to Weaviate...")
        self.client = weaviate.connect_to_wcs(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )

        # Create collection
        self.collection_name = "FashionProducts"
        self._create_collection()

    def _create_collection(self):
        """Create Weaviate collection for fashion products"""
        try:
            # Delete existing collection if it exists
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")

            # Create new collection
            self.collection = self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own vectors
                properties=[
                    weaviate.classes.config.Property(name="product_id",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="name", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="price", data_type=weaviate.classes.config.DataType.NUMBER),
                    weaviate.classes.config.Property(name="colour", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="brand", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="img_url", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="rating_count",
                                                     data_type=weaviate.classes.config.DataType.NUMBER),
                    weaviate.classes.config.Property(name="avg_rating",
                                                     data_type=weaviate.classes.config.DataType.NUMBER),
                    weaviate.classes.config.Property(name="description",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="attributes",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="local_image_path",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                ]
            )
            print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Get existing collection
            self.collection = self.client.collections.get(self.collection_name)

    def generate_image_embedding(self, image_path_or_url):
        """
        Generate CLIP embedding for an image

        Args:
            image_path_or_url: Local path or URL to the image

        Returns:
            numpy array of embedding
        """
        try:
            if image_path_or_url.startswith('http'):
                # Load from URL
                response = requests.get(image_path_or_url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                image = Image.open(image_path_or_url).convert('RGB')

            # Process image and generate embedding
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error generating embedding for {image_path_or_url}: {e}")
            return None

    def load_and_index_data(self, csv_path, images_folder_path):
        """
        Load data from CSV and index in Weaviate

        Args:
            csv_path: Path to the CSV file
            images_folder_path: Path to the images folder
        """
        print("Loading CSV data...")
        df = pd.read_csv(csv_path)

        print(f"Processing {len(df)} products...")

        # Process each product
        successful_insertions = 0
        failed_insertions = 0

        for idx, row in df.iterrows():
            try:
                # Construct local image path
                local_image_path = os.path.join(images_folder_path, f"{idx}.jpg")

                # Check if local image exists, otherwise use URL
                if os.path.exists(local_image_path):
                    embedding = self.generate_image_embedding(local_image_path)
                    image_source = local_image_path
                else:
                    # Try with URL
                    embedding = self.generate_image_embedding(row['img'])
                    image_source = row['img']
                    local_image_path = "N/A"

                if embedding is None:
                    failed_insertions += 1
                    continue

                # Prepare data for Weaviate
                data_object = {
                    "product_id": str(row['p_id']) if pd.notna(row['p_id']) else str(idx),
                    "name": str(row['name']) if pd.notna(row['name']) else "",
                    "price": float(row['price']) if pd.notna(row['price']) else 0.0,
                    "colour": str(row['colour']) if pd.notna(row['colour']) else "",
                    "brand": str(row['brand']) if pd.notna(row['brand']) else "",
                    "img_url": str(row['img']) if pd.notna(row['img']) else "",
                    "rating_count": float(row['ratingCount']) if pd.notna(row['ratingCount']) else 0.0,
                    "avg_rating": float(row['avg_rating']) if pd.notna(row['avg_rating']) else 0.0,
                    "description": str(row['description']) if pd.notna(row['description']) else "",
                    "attributes": str(row['p_attributes']) if pd.notna(row['p_attributes']) else "",
                    "local_image_path": local_image_path
                }

                # Insert into Weaviate
                self.collection.data.insert(
                    properties=data_object,
                    vector=embedding.tolist()
                )

                successful_insertions += 1

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1} products...")

            except Exception as e:
                print(f"Error processing product {idx}: {e}")
                failed_insertions += 1
                continue

        print(f"Indexing complete!")
        print(f"Successful insertions: {successful_insertions}")
        print(f"Failed insertions: {failed_insertions}")

    def search_similar_products(self, query_image_path, top_k=5):
        """
        Search for similar products using an image query

        Args:
            query_image_path: Path to query image
            top_k: Number of similar products to return

        Returns:
            List of similar products with metadata
        """
        print(f"Searching for products similar to: {query_image_path}")

        # Generate embedding for query image
        query_embedding = self.generate_image_embedding(query_image_path)
        if query_embedding is None:
            print("Failed to generate embedding for query image")
            return []

        try:
            # Perform vector search
            results = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=top_k,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )

            similar_products = []
            for result in results.objects:
                product = {
                    'product_id': result.properties['product_id'],
                    'name': result.properties['name'],
                    'price': result.properties['price'],
                    'colour': result.properties['colour'],
                    'brand': result.properties['brand'],
                    'img_url': result.properties['img_url'],
                    'rating_count': result.properties['rating_count'],
                    'avg_rating': result.properties['avg_rating'],
                    'description': result.properties['description'][:200] + "..." if len(
                        result.properties['description']) > 200 else result.properties['description'],
                    'local_image_path': result.properties['local_image_path'],
                    'similarity_score': 1 - result.metadata.distance  # Convert distance to similarity
                }
                similar_products.append(product)

            return similar_products

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def display_results(self, results):
        """Display search results in a formatted way"""
        if not results:
            print("No results found!")
            return

        print(f"\n{'=' * 100}")
        print(f"FOUND {len(results)} SIMILAR PRODUCTS")
        print(f"{'=' * 100}")

        for i, product in enumerate(results, 1):
            print(f"\n{'-' * 50} RESULT {i} {'-' * 50}")
            print(f"Product ID: {product['product_id']}")
            print(f"Name: {product['name']}")
            print(f"Brand: {product['brand']}")
            print(f"Price: ₹{product['price']}")
            print(f"Color: {product['colour']}")
            print(f"Rating: {product['avg_rating']:.2f} ({int(product['rating_count'])} reviews)")
            print(f"Similarity Score: {product['similarity_score']:.4f}")
            print(f"Image URL: {product['img_url']}")
            print(f"Local Image: {product['local_image_path']}")
            print(f"Description: {product['description']}")
            print(f"{'-' * 100}")

    def close(self):
        """Close Weaviate connection"""
        self.client.close()
        print("Connection closed.")


# Usage example
def main():
    # Configuration - REPLACE WITH YOUR ACTUAL CREDENTIALS
    WEAVIATE_URL = "_url"  # Replace with your cluster URL
    WEAVIATE_API_KEY = "_API_key"  # Replace with your API key

    # File paths - UPDATE THESE PATHS
    CSV_PATH = "Fashion Dataset.csv"  # Path to your CSV file
    IMAGES_FOLDER = "images"  # Path to your images folder

    try:
        # Initialize search engine
        search_engine = FashionSearchEngine(WEAVIATE_URL, WEAVIATE_API_KEY)

        # Load and index data
        search_engine.load_and_index_data(CSV_PATH, IMAGES_FOLDER)

        # Test search with a sample image
        query_image = os.path.join(".", "img.jpg")  # Use first image as query

        if os.path.exists(query_image):
            print(f"\nTesting search with query image: {query_image}")
            results = search_engine.search_similar_products(query_image, top_k=5)
            search_engine.display_results(results)
        else:
            print(f"Query image not found: {query_image}")
            print("Make sure your images are in the correct folder with naming convention: 0.jpg, 1.jpg, etc.")

        # Close connection
        search_engine.close()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Replace WEAVIATE_URL and WEAVIATE_API_KEY with your actual credentials")
        print("2. Update CSV_PATH and IMAGES_FOLDER with correct paths")
        print("3. Ensure your CSV file and images folder exist")


if __name__ == "__main__":
    main()