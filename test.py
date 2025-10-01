import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
import os
import weaviate
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")


class SimpleFashionSearch:
    def __init__(self, weaviate_url, weaviate_api_key):
        """Initialize search engine with existing database"""
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)

        print("Loading text embedding model...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Connecting to Weaviate...")
        self.client = weaviate.connect_to_wcs(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )

        # Connect to existing collection
        self.collection_name = "EnhancedFashionProducts"
        self.collection = self.client.collections.get(self.collection_name)
        print(f"Connected to existing collection: {self.collection_name}")

    def generate_image_embedding(self, image_path):
        """Generate CLIP embedding for query image"""
        try:
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')

            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None

    def generate_text_embedding(self, text):
        """Generate text embedding"""
        try:
            embedding = self.text_model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return None

    def search_similar_products(self, query_image_path, search_mode="combined",
                                style_filter=None, color_filter=None, top_k=5):
        """Search for similar products"""
        print(f"\nSearching for products similar to: {query_image_path}")
        print(f"Mode: {search_mode}, Style filter: {style_filter}, Color filter: {color_filter}")

        # Generate query image embedding
        query_image_embedding = self.generate_image_embedding(query_image_path)
        if query_image_embedding is None:
            print("Failed to generate embedding for query image")
            return []

        try:
            # Prepare filters
            where_filter = None
            if style_filter or color_filter:
                conditions = []
                if style_filter:
                    conditions.append({"path": "style_category", "operator": "Equal", "valueText": style_filter})
                if color_filter:
                    conditions.append({"path": "colour", "operator": "Like", "valueText": f"*{color_filter}*"})

                if len(conditions) == 1:
                    where_filter = conditions[0]
                else:
                    where_filter = {"operator": "And", "operands": conditions}

            # Generate search vector based on mode
            if search_mode == "image":
                search_vector = query_image_embedding.tolist()
            elif search_mode == "combined":
                # Create combined vector (same as during indexing)
                search_text = "fashion clothing item"
                text_embedding = self.generate_text_embedding(search_text)

                # Match dimensions
                img_dim = len(query_image_embedding)
                txt_dim = len(text_embedding)
                if txt_dim != img_dim:
                    if txt_dim > img_dim:
                        text_embedding = text_embedding[:img_dim]
                    else:
                        text_embedding = np.pad(text_embedding, (0, img_dim - txt_dim), mode='constant')

                # Weighted combination
                image_weight = 0.8
                text_weight = 0.2
                combined_embedding = (image_weight * query_image_embedding + text_weight * text_embedding)
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                search_vector = combined_embedding.tolist()
            else:
                search_vector = query_image_embedding.tolist()

            # Search in Weaviate
            if where_filter:
                results = self.collection.query.near_vector(
                    near_vector=search_vector,
                    limit=top_k,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                ).where(where_filter)
            else:
                results = self.collection.query.near_vector(
                    near_vector=search_vector,
                    limit=top_k,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )

            return self._format_results(results.objects)

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def _format_results(self, results):
        """Format search results"""
        similar_products = []
        for result in results:
            product = {
                'product_id': result.properties['product_id'],
                'name': result.properties['name'],
                'price': result.properties['price'],
                'colour': result.properties['colour'],
                'brand': result.properties['brand'],
                'img_url': result.properties['img_url'],
                'rating_count': result.properties['rating_count'],
                'avg_rating': result.properties['avg_rating'],
                'description': result.properties['description'][:100] + "..." if len(
                    result.properties['description']) > 100 else result.properties['description'],
                'local_image_path': result.properties['local_image_path'],
                'clothing_type': result.properties.get('clothing_type', 'unknown'),
                'style_category': result.properties.get('style_category', 'unknown'),
                'similarity_score': 1 - result.metadata.distance
            }
            similar_products.append(product)
        return similar_products

    def display_results(self, results):
        """Display search results"""
        if not results:
            print("No results found!")
            return

        print(f"\n{'=' * 100}")
        print(f"FOUND {len(results)} SIMILAR PRODUCTS")
        print(f"{'=' * 100}")

        for i, product in enumerate(results, 1):
            print(f"\n--- RESULT {i} ---")
            print(f"Name: {product['name']}")
            print(f"Brand: {product['brand']}")
            print(f"Price: ₹{product['price']}")
            print(f"Color: {product['colour']}")
            print(f"Type: {product['clothing_type']}")
            print(f"Style: {product['style_category']}")
            print(f"Rating: {product['avg_rating']:.1f}/5 ({int(product['rating_count'])} reviews)")
            print(f"Similarity: {product['similarity_score']:.3f}")
            print(f"Image: {product['local_image_path']}")
            print(f"Description: {product['description']}")
            print("-" * 80)

    def close(self):
        """Close connection"""
        self.client.close()
        print("Connection closed.")


def main():
    # Your Weaviate configuration
    WEAVIATE_URL = "_url"
    WEAVIATE_API_KEY = "_api_key"

    # Query image path
    QUERY_IMAGE = "img_1.png"  # Change this to your image path

    try:
        # Initialize search engine
        search_engine = SimpleFashionSearch(WEAVIATE_URL, WEAVIATE_API_KEY)

        # Check if query image exists
        if not os.path.exists(QUERY_IMAGE):
            print(f"Error: Query image '{QUERY_IMAGE}' not found!")
            print("Please make sure your image file exists in the current directory.")
            return

        print("\n" + "=" * 60)
        print("FASHION SIMILARITY SEARCH")
        print("=" * 60)

        # 1. Basic image search
        print(f"\n🔍 BASIC IMAGE SEARCH")
        results = search_engine.search_similar_products(
            QUERY_IMAGE,
            search_mode="image",
            top_k=5
        )
        search_engine.display_results(results)

        # 2. Enhanced combined search
        print(f"\n🔍 ENHANCED COMBINED SEARCH")
        results = search_engine.search_similar_products(
            QUERY_IMAGE,
            search_mode="combined",
            top_k=5
        )
        search_engine.display_results(results)

        # 3. Filtered search - ethnic only
        print(f"\n🔍 ETHNIC CLOTHING ONLY")
        results = search_engine.search_similar_products(
            QUERY_IMAGE,
            search_mode="combined",
            style_filter="ethnic",
            top_k=3
        )
        search_engine.display_results(results)

        # 4. Filtered search - western only
        print(f"\n🔍 WESTERN CLOTHING ONLY")
        results = search_engine.search_similar_products(
            QUERY_IMAGE,
            search_mode="combined",
            style_filter="western",
            top_k=3
        )
        search_engine.display_results(results)

        # Close connection
        search_engine.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()