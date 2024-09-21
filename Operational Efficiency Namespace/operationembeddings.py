import logging
import numpy as np
import pinecone
import snowflake.connector
from tqdm import tqdm

# Step 1: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Step 2: Fetch data from Snowflake
def fetch_data_from_snowflake():
    try:
        logger.info("Connecting to Snowflake...")
        # Snowflake connection details
        conn = snowflake.connector.connect(
            user='OPSCALEAI',
            password='Opscale2030',
            account='nvvmnod-mw08757',
            warehouse='DIANA_DATA_LAKE',
            database='DIANA_SALES_ES',
            schema='SALES'
        )
        cur = conn.cursor()

        # SQL query to aggregate store transaction data and join with store locations
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

# Step 3: Initialize Pinecone
def initialize_pinecone():
    try:
        logger.info("Initializing Pinecone...")
        pinecone.init(api_key="your-pinecone-api-key")

        index_name = "operational-efficiency"
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating Pinecone index '{index_name}'...")
            pinecone.create_index(index_name, dimension=4)  # Adjust dimensions based on your features
        logger.info(f"Connecting to Pinecone index '{index_name}'...")
        return pinecone.Index(index_name)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise

# Step 4: Vectorization with Logistics Data
def vectorize_logistics_data(store_logistics_data):
    logger.info("Vectorizing logistics data...")
    vectorized_logistics = []

    for row in tqdm(store_logistics_data, desc="Vectorizing data"):
        store_id, store_name, total_transactions, avg_transaction_value, total_quantity, address, region_id = row

        # Create a vector for the store's logistics and transaction patterns
        vector = np.array([
            total_transactions,
            avg_transaction_value,
            total_quantity,
            region_id  # Assuming region_id can be used as a feature
        ])

        # Add metadata for store location and additional data
        vectorized_logistics.append({
            'id': str(store_id),
            'values': vector.tolist(),
            'metadata': {
                'store_name': store_name,
                'address': address,
                'region_id': region_id
            }
        })

    logger.info("Finished vectorizing logistics data.")
    return vectorized_logistics

# Step 5: Batch Upload to Pinecone
def batch_upload_to_pinecone(index, vectorized_logistics, batch_size=100):
    logger.info("Uploading vectors to Pinecone in batches...")
    
    # Progress bar for batch upload
    for i in tqdm(range(0, len(vectorized_logistics), batch_size), desc="Uploading to Pinecone"):
        batch = vectorized_logistics[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            logger.error(f"Error uploading batch {i // batch_size + 1}: {e}")
            raise

    logger.info("Successfully uploaded all vectors to Pinecone.")

# Main process
def main():
    try:
        # Step 1: Fetch data from Snowflake
        store_logistics_data = fetch_data_from_snowflake()

        # Step 2: Initialize Pinecone and create or connect to the index
        pinecone_index = initialize_pinecone()

        # Step 3: Vectorize and prepare data for Pinecone
        vectorized_logistics = vectorize_logistics_data(store_logistics_data)

        # Step 4: Batch upload the data to Pinecone
        batch_upload_to_pinecone(pinecone_index, vectorized_logistics)

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()
