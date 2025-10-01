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
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


class EnhancedFashionSearchEngine:
    def __init__(self, weaviate_url, weaviate_api_key):
        """
        Initialize the Enhanced Fashion Search Engine with multimodal search capabilities
        """
        # Initialize CLIP model for images
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)

        # Initialize sentence transformer for text features
        print("Loading text embedding model...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Weaviate client
        print("Connecting to Weaviate...")
        self.client = weaviate.connect_to_wcs(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )

        # Create collection
        self.collection_name = "EnhancedFashionProducts"
        self._create_collection()

    def _create_collection(self):
        """Create enhanced Weaviate collection with single combined vector"""
        try:
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")

            # Create collection with single vector (combined approach)
            self.collection = self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
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
                    weaviate.classes.config.Property(name="clothing_type",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="style_category",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                    # Store individual scores as properties for advanced filtering
                    weaviate.classes.config.Property(name="image_vector_str",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="text_vector_str",
                                                     data_type=weaviate.classes.config.DataType.TEXT),
                ]
            )
            print(f"Created enhanced collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
            self.collection = self.client.collections.get(self.collection_name)

    def generate_image_embedding(self, image_path_or_url):
        """Generate CLIP embedding for an image"""
        try:
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')

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
        """Generate text embedding using sentence transformer"""
        try:
            embedding = self.text_model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return None

    def extract_clothing_features(self, name, description, attributes):
        """Extract clothing type and style from text data"""
        text_content = f"{name} {description} {attributes}".lower()

        # Define clothing types
        clothing_types = {
            'kurta': ['kurta', 'kurti', 'kurtis'],
            'dress': ['dress', 'gown', 'frock'],
            'top': ['top', 'blouse', 'shirt', 'tee'],
            'bottom': ['pant', 'jeans', 'trouser', 'legging'],
            'ethnic': ['ethnic', 'traditional', 'indian'],
            'western': ['western', 'casual', 'formal'],
            'winter': ['winter', 'warm', 'fur', 'poncho', 'coat', 'jacket'],
            'saree': ['saree', 'sari'],
            'salwar': ['salwar', 'palazzo', 'churidar']
        }

        # Determine clothing type
        detected_types = []
        for category, keywords in clothing_types.items():
            if any(keyword in text_content for keyword in keywords):
                detected_types.append(category)

        clothing_type = ', '.join(detected_types) if detected_types else 'unknown'

        # Determine style category
        if any(word in text_content for word in
               ['ethnic', 'traditional', 'indian', 'kurta', 'kurti', 'saree', 'salwar']):
            style_category = 'ethnic'
        elif any(word in text_content for word in ['western', 'dress', 'jeans', 'casual', 'formal']):
            style_category = 'western'
        else:
            style_category = 'unknown'

        return clothing_type, style_category

    def load_and_index_data(self, csv_path, images_folder_path):
        """Load and index data with enhanced features"""
        print("Loading CSV data...")
        df = pd.read_csv(csv_path)
        print(f"Processing {len(df)} products...")

        successful_insertions = 0
        failed_insertions = 0

        for idx, row in df.iterrows():
            try:
                # Construct paths
                local_image_path = os.path.join(images_folder_path, f"{idx}.jpg")

                # Generate image embedding
                if os.path.exists(local_image_path):
                    image_embedding = self.generate_image_embedding(local_image_path)
                    image_source = local_image_path
                else:
                    image_embedding = self.generate_image_embedding(row['img'])
                    image_source = row['img']
                    local_image_path = "N/A"

                if image_embedding is None:
                    failed_insertions += 1
                    continue

                # Prepare text for embedding
                name = str(row['name']) if pd.notna(row['name']) else ""
                description = str(row['description']) if pd.notna(row['description']) else ""
                attributes = str(row['p_attributes']) if pd.notna(row['p_attributes']) else ""
                colour = str(row['colour']) if pd.notna(row['colour']) else ""
                brand = str(row['brand']) if pd.notna(row['brand']) else ""

                # Extract clothing features
                clothing_type, style_category = self.extract_clothing_features(name, description, attributes)

                # Create comprehensive text description
                text_description = f"Product: {name} Brand: {brand} Color: {colour} Type: {clothing_type} Style: {style_category} Description: {description} Attributes: {attributes}"

                # Generate text embedding
                text_embedding = self.generate_text_embedding(text_description)
                if text_embedding is None:
                    failed_insertions += 1
                    continue

                # Create combined embedding (weighted combination)
                image_weight = 0.7
                text_weight = 0.3

                # Ensure both embeddings have same dimension by padding/truncating
                img_dim = len(image_embedding)
                txt_dim = len(text_embedding)

                if img_dim != txt_dim:
                    # Resize text embedding to match image embedding dimension
                    if txt_dim > img_dim:
                        text_embedding = text_embedding[:img_dim]
                    else:
                        # Pad with zeros
                        text_embedding = np.pad(text_embedding, (0, img_dim - txt_dim), mode='constant')

                combined_embedding = (image_weight * image_embedding + text_weight * text_embedding)
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)  # Normalize

                # Prepare data object
                data_object = {
                    "product_id": str(row['p_id']) if pd.notna(row['p_id']) else str(idx),
                    "name": name,
                    "price": float(row['price']) if pd.notna(row['price']) else 0.0,
                    "colour": colour,
                    "brand": brand,
                    "img_url": str(row['img']) if pd.notna(row['img']) else "",
                    "rating_count": float(row['ratingCount']) if pd.notna(row['ratingCount']) else 0.0,
                    "avg_rating": float(row['avg_rating']) if pd.notna(row['avg_rating']) else 0.0,
                    "description": description,
                    "attributes": attributes,
                    "local_image_path": local_image_path,
                    "clothing_type": clothing_type,
                    "style_category": style_category
                }

                # Add vector strings as properties for potential future use
                data_object["image_vector_str"] = str(image_embedding.tolist())
                data_object["text_vector_str"] = str(text_embedding.tolist())

                # Insert with single combined vector
                self.collection.data.insert(
                    properties=data_object,
                    vector=combined_embedding.tolist()
                )

                successful_insertions += 1

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1} products...")

            except Exception as e:
                print(f"Error processing product {idx}: {e}")
                failed_insertions += 1
                continue

        print(f"Enhanced indexing complete!")
        print(f"Successful insertions: {successful_insertions}")
        print(f"Failed insertions: {failed_insertions}")

    def search_similar_products(self, query_image_path, search_mode="combined",
                                style_filter=None, color_filter=None, top_k=5):
        """
        Enhanced search with filtering capabilities

        Args:
            query_image_path: Path to query image
            search_mode: "image", "text", or "combined"
            style_filter: "ethnic", "western", or None
            color_filter: Color name or None
            top_k: Number of results to return
        """
        print(f"Searching with mode: {search_mode}, style_filter: {style_filter}, color_filter: {color_filter}")

        # Generate query embeddings
        query_image_embedding = self.generate_image_embedding(query_image_path)
        if query_image_embedding is None:
            print("Failed to generate embedding for query image")
            return []

        try:
            # Prepare where filter
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
                # Use only image features for search vector
                search_vector = query_image_embedding.tolist()
            elif search_mode == "text":
                # Generate text embedding from image (could be enhanced with image captioning)
                # For now, create a generic search text from the image
                search_text = "black clothing item fashion product"  # This could be improved with image captioning
                text_embedding = self.generate_text_embedding(search_text)

                # Resize to match image embedding dimension
                img_dim = len(query_image_embedding)
                txt_dim = len(text_embedding)
                if txt_dim != img_dim:
                    if txt_dim > img_dim:
                        text_embedding = text_embedding[:img_dim]
                    else:
                        text_embedding = np.pad(text_embedding, (0, img_dim - txt_dim), mode='constant')

                search_vector = text_embedding.tolist()
            else:  # combined
                # Create combined vector similar to indexing
                search_text = "fashion clothing item"  # Generic text for query
                text_embedding = self.generate_text_embedding(search_text)

                # Ensure same dimensions
                img_dim = len(query_image_embedding)
                txt_dim = len(text_embedding)
                if txt_dim != img_dim:
                    if txt_dim > img_dim:
                        text_embedding = text_embedding[:img_dim]
                    else:
                        text_embedding = np.pad(text_embedding, (0, img_dim - txt_dim), mode='constant')

                # Weighted combination (favor image more for visual queries)
                image_weight = 0.8
                text_weight = 0.2
                combined_embedding = (image_weight * query_image_embedding + text_weight * text_embedding)
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                search_vector = combined_embedding.tolist()

            # Perform search
            results = self.collection.query.near_vector(
                near_vector=search_vector,
                limit=top_k,
                where=where_filter,
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
                'description': result.properties['description'][:200] + "..." if len(
                    result.properties['description']) > 200 else result.properties['description'],
                'local_image_path': result.properties['local_image_path'],
                'clothing_type': result.properties.get('clothing_type', 'unknown'),
                'style_category': result.properties.get('style_category', 'unknown'),
                'similarity_score': 1 - result.metadata.distance
            }
            similar_products.append(product)

        return similar_products

    def display_results(self, results):
        """Display enhanced search results"""
        if not results:
            print("No results found!")
            return

        print(f"\n{'=' * 120}")
        print(f"FOUND {len(results)} SIMILAR PRODUCTS")
        print(f"{'=' * 120}")

        for i, product in enumerate(results, 1):
            print(f"\n{'-' * 50} RESULT {i} {'-' * 50}")
            print(f"Product ID: {product['product_id']}")
            print(f"Name: {product['name']}")
            print(f"Brand: {product['brand']}")
            print(f"Price: ₹{product['price']}")
            print(f"Color: {product['colour']}")
            print(f"Clothing Type: {product['clothing_type']}")
            print(f"Style Category: {product['style_category']}")
            print(f"Rating: {product['avg_rating']:.2f} ({int(product['rating_count'])} reviews)")
            print(f"Similarity Score: {product['similarity_score']:.4f}")
            print(f"Image URL: {product['img_url']}")
            print(f"Local Image: {product['local_image_path']}")
            print(f"Description: {product['description']}")
            print(f"{'-' * 120}")

    def close(self):
        """Close connections"""
        self.client.close()
        print("Connection closed.")


# Usage example with different search modes
def main():
    # Configuration
    WEAVIATE_URL = "_url"
    WEAVIATE_API_KEY = "_Api_Key"
    CSV_PATH = "Fashion Dataset.csv"
    IMAGES_FOLDER = "images"

    try:
        # Initialize enhanced search engine
        search_engine = EnhancedFashionSearchEngine(WEAVIATE_URL, WEAVIATE_API_KEY)

        # Load and index data with enhanced features
        search_engine.load_and_index_data(CSV_PATH, IMAGES_FOLDER)

        # Test different search modes
        query_image = "img.jpg"

        if os.path.exists(query_image):
            print("\n" + "=" * 80)
            print("TESTING DIFFERENT SEARCH MODES")
            print("=" * 80)

            # 1. Image-only search
            print("\n1. IMAGE-ONLY SEARCH:")
            results = search_engine.search_similar_products(query_image, search_mode="image", top_k=3)
            search_engine.display_results(results)

            # 2. Combined search (image + text features)
            print("\n2. COMBINED SEARCH (Image + Text):")
            results = search_engine.search_similar_products(query_image, search_mode="combined", top_k=3)
            search_engine.display_results(results)

            # 3. Filtered search - only ethnic style
            print("\n3. FILTERED SEARCH (Ethnic style only):")
            results = search_engine.search_similar_products(
                query_image, search_mode="combined", style_filter="ethnic", top_k=3
            )
            search_engine.display_results(results)

            # 4. Color-filtered search
            print("\n4. COLOR-FILTERED SEARCH (Black items only):")
            results = search_engine.search_similar_products(
                query_image, search_mode="combined", color_filter="black", top_k=3
            )
            search_engine.display_results(results)

        search_engine.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()