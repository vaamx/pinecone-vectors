import snowflake.connector
from sentence_transformers import SentenceTransformer
import numpy as np
import pinecone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging to monitor the workflow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Securely connect and query product data from Snowflake

def query_product_data():
    try:
        # Establish a connection to Snowflake
        conn = snowflake.connector.connect(
            user='OPSCALEAI',
            password='Opscale2030',  # Consider using a secure vault or environment variable for the password
            account='nvvmnod-mw08757',
            warehouse='DIANA_DATA_LAKE',
            database='DIANA_SALES_ES',
            schema='SALES'
        )

        cur = conn.cursor()

        # SQL query to fetch product details
        query = """
        SELECT PRODUCT_ID, PRODUCT_NAME, DESCRIPTION, PRICE, PRODUCT_CATEGORY_ID
        FROM DIANA_SALES_ES.SALES.PRODUCTS
        """

        cur.execute(query)
        product_data = cur.fetchall()

        # Log the number of products retrieved
        logging.info(f"Retrieved {len(product_data)} products from Snowflake.")

        cur.close()
        conn.close()

        return product_data

    except Exception as e:
        logging.error(f"Error querying Snowflake: {e}")
        return []


# Step 2: Optimize Vectorization (Parallel Processing with Pretrained Embedding Model)

# Load a pretrained model for embeddings (e.g., 'all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

def vectorize_description(description):
    return model.encode(description).tolist()

# Parallelize the vectorization process to handle large datasets efficiently
def vectorize_products_parallel(product_data):
    vectorized_products = []

    def process_row(row):
        product_id, product_name, description, price, product_category_id = row
        vector = vectorize_description(description)

        # Add metadata for better filtering
        metadata = {
            'product_name': product_name,
            'price': price,
            'product_category_id': product_category_id,
            'description_length': len(description),  # Useful for future search optimizations
        }

        return {'id': str(product_id), 'values': vector, 'metadata': metadata}

    # Use ThreadPoolExecutor to parallelize vectorization
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row) for row in product_data]
        
        # Add a progress bar for vectorization
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vectorizing Products", unit="product"):
            try:
                vectorized_products.append(future.result())
            except Exception as e:
                logging.error(f"Error in vectorization: {e}")

    logging.info(f"Vectorized {len(vectorized_products)} products.")
    return vectorized_products


# Step 3: Efficiently Upload to Pinecone (Batch Processing with Progress Bars)

def upload_to_pinecone(vectorized_products):
    try:
        # Initialize Pinecone
        pinecone.init(api_key="your-pinecone-api-key")

        # Create or connect to an index
        index_name = "product-recommendations"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384)  # Adjust dimension based on embedding model output

        index = pinecone.Index(index_name)

        # Upload data in batches to Pinecone
        batch_size = 100
        num_batches = len(vectorized_products) // batch_size + 1
        
        # Add a progress bar for batch uploads
        for i in tqdm(range(0, len(vectorized_products), batch_size), desc="Uploading to Pinecone", unit="batch"):
            batch = vectorized_products[i:i + batch_size]
            index.upsert(vectors=batch)
            logging.info(f"Uploaded batch {i // batch_size + 1} of {num_batches} to Pinecone.")

    except Exception as e:
        logging.error(f"Error uploading to Pinecone: {e}")


# Main Function: Execute the Workflow

def main():
    # Step 1: Query product data from Snowflake
    product_data = query_product_data()

    if product_data:
        # Step 2: Vectorize product descriptions
        vectorized_products = vectorize_products_parallel(product_data)

        # Step 3: Upload vectorized data to Pinecone
        upload_to_pinecone(vectorized_products)


if __name__ == "__main__":
    main()
