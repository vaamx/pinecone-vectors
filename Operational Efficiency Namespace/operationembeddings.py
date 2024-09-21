import os
import logging
import numpy as np
import pinecone
import snowflake.connector
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import openai

# Step 1: Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
snowflake_user = os.getenv("SNOWFLAKE_USER")
snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
snowflake_database = os.getenv("SNOWFLAKE_DATABASE")
snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")

# Step 2: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Step 3: Initialize Pinecone
def initialize_pinecone(index_name="diana-sales", embedding_dimension=1536):
    try:
        logger.info("Initializing Pinecone...")
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
        logger.info(f"Connecting to Pinecone index '{index_name}'...")
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise

# Step 4: Fetch data from Snowflake
def fetch_data_from_snowflake():
    try:
        logger.info("Connecting to Snowflake...")
        conn = snowflake.connector.connect(
            user=snowflake_user,
            password=snowflake_password,
            account=snowflake_account,
            warehouse=snowflake_warehouse,
            database=snowflake_database,
            schema=snowflake_schema
        )
        cur = conn.cursor()

        query = """
        SELECT st.STORE_ID, st.STORE_NAME, 
               COUNT(t.TRANSACTION_ID) AS total_transactions, 
               AVG(t.TOTAL_VALUE) AS avg_transaction_value, 
               SUM(t.QUANTITY) AS total_quantity, 
               st.ADDRESS, st.REGION_ID
        FROM DIANA_SALES_ES.SALES.TRANSACTIONS t
        JOIN DIANA_SALES_ES.STOREFRONTS.STORES st ON t.STORE_ID = st.STORE_ID
        GROUP BY st.STORE_ID, st.STORE_NAME, st.ADDRESS, st.REGION_ID
        """
        
        logger.info("Executing query on Snowflake...")
        cur.execute(query)
        store_logistics_data = cur.fetchall()
        logger.info(f"Fetched {len(store_logistics_data)} records from Snowflake.")
        return store_logistics_data
    except Exception as e:
        logger.error(f"Error fetching data from Snowflake: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# Step 5: Vectorization function for each row
def process_row(row, embedding_dimension=1536):
    store_id, store_name, total_transactions, avg_transaction_value, total_quantity, address, region_id = row
    
    # Create the vector, leaving unused elements as zeros for now
    vector = np.zeros(embedding_dimension)
    vector[:4] = np.array([total_transactions, avg_transaction_value, total_quantity, region_id])
    
    # Add metadata
    metadata = {
        'store_name': store_name,
        'address': address,
        'region_id': region_id
    }
    
    return {
        'id': str(store_id),
        'values': vector.tolist(),
        'metadata': metadata
    }

# Step 6: Vectorize all logistics data using concurrency
def vectorize_logistics_data(store_logistics_data, embedding_dimension=1536):
    vectorized_logistics = []
    logger.info("Vectorizing logistics data with concurrency...")
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row, embedding_dimension) for row in store_logistics_data]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vectorizing Sales Data", unit="record"):
            try:
                result = future.result()
                if result:
                    vectorized_logistics.append(result)
            except Exception as e:
                logger.error(f"Error in vectorization: {e}")
    
    logger.info(f"Finished vectorizing {len(vectorized_logistics)} records.")
    return vectorized_logistics

# Step 7: Batch upload to Pinecone with namespace handling
def batch_upload_to_pinecone(index, vectorized_logistics, namespace="logistics", batch_size=100):
    logger.info(f"Uploading vectors to Pinecone in namespace '{namespace}'...")
    
    for i in tqdm(range(0, len(vectorized_logistics), batch_size), desc="Uploading to Pinecone", unit="batch"):
        batch = vectorized_logistics[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            logger.error(f"Error during Pinecone upsert: {e}")
            raise
    
    logger.info("Successfully uploaded all vectors to Pinecone.")

# Step 8: Main process
def main():
    try:
        # Fetch data from Snowflake
        store_logistics_data = fetch_data_from_snowflake()

        # Initialize Pinecone and connect to the index
        pinecone_index = initialize_pinecone()

        # Vectorize the logistics data
        vectorized_logistics = vectorize_logistics_data(store_logistics_data)

        # Batch upload the vectorized data to Pinecone using a namespace
        batch_upload_to_pinecone(pinecone_index, vectorized_logistics, namespace="logistics")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()
