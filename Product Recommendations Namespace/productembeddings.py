import os
import snowflake.connector
import openai
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load sensitive keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Step 1: Query product data from Snowflake securely
def query_product_data():
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )

        cur = conn.cursor()

        # SQL query to fetch product details
        query = """
        SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICE, PRODUCT_CATEGORY_ID
        FROM PRODUCTS
        """

        cur.execute(query)
        product_data = cur.fetchall()

        logging.info(f"Retrieved {len(product_data)} products from Snowflake.")
        cur.close()
        conn.close()

        return product_data

    except Exception as e:
        logging.error(f"Error querying Snowflake: {e}")
        return []

# Step 2: Use OpenAI API for 1536-Dimensional Embeddings (Corrected API Usage)
def vectorize_description(description):
    try:
        # Correct API call for OpenAI embedding
        response = openai.Embedding.create(
            input=description,  # input should be a single string
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {e}")
        return None

# Parallelize the vectorization process to handle large datasets efficiently
def vectorize_products_parallel(product_data):
    vectorized_products = []

    def process_row(row):
        product_id, product_name, description, price, product_category_id = row
        vector = vectorize_description(description)

        if vector:  # Proceed only if vectorization succeeded
            metadata = {
                'product_name': product_name,
                'price': float(price),  # Convert Decimal to float
                'product_category_id': product_category_id,
                'description_length': len(description),  # Useful for search optimizations
            }
            return {'id': str(product_id), 'values': vector, 'metadata': metadata}
        return None  # Return None if vectorization failed

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row) for row in product_data]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vectorizing Products", unit="product"):
            try:
                result = future.result()
                if result:
                    vectorized_products.append(result)
            except Exception as e:
                logging.error(f"Error in vectorization: {e}")

    logging.info(f"Vectorized {len(vectorized_products)} products.")
    return vectorized_products

# Step 3: Efficiently Upload to Pinecone (Batch Processing with Progress Bars)
def upload_to_pinecone(vectorized_products, namespace="default_namespace"):
    try:
        # Instantiate Pinecone
        pc = Pinecone(api_key=pinecone_api_key)

        index_name = "diana-sales"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # Ensure dimensional consistency with OpenAI embeddings
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )

        index = pc.Index(index_name)

        # Batch upload
        batch_size = 100
        for i in tqdm(range(0, len(vectorized_products), batch_size), desc="Uploading to Pinecone", unit="batch"):
            batch = vectorized_products[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            logging.info(f"Uploaded batch {i // batch_size + 1} to Pinecone.")

    except Exception as e:
        logging.error(f"Error uploading to Pinecone: {e}")

# Main Function: Execute the Workflow
def main(namespace="default_namespace"):
    product_data = query_product_data()

    if product_data:
        vectorized_products = vectorize_products_parallel(product_data)

        if vectorized_products:
            upload_to_pinecone(vectorized_products, namespace=namespace)
        else:
            logging.error("No vectorized products to upload.")
    else:
        logging.error("No product data retrieved.")

if __name__ == "__main__":
    main(namespace="product_descriptions_2024")
