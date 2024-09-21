import os
import logging
from tqdm import tqdm
import numpy as np
import pinecone
import snowflake.connector

# Step 0: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Fetch Sales Data from Snowflake
def fetch_sales_data():
    logging.info("Connecting to Snowflake...")
    try:
        # Connect to Snowflake (consider using environment variables for security)
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER', 'OPSCALEAI'),
            password=os.getenv('SNOWFLAKE_PASSWORD', 'Opscale2030'),
            account=os.getenv('SNOWFLAKE_ACCOUNT', 'nvvmnod-mw08757'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'DIANA_DATA_LAKE'),
            database=os.getenv('SNOWFLAKE_DATABASE', 'DIANA_SALES_ES'),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'SALES')
        )
        cur = conn.cursor()

        logging.info("Executing sales data query...")
        # SQL query to aggregate sales data by salesperson and region
        query = """
        SELECT s.SALESREP_ID, s.SALESREP_NAME, sp.REGION_ID,
               SUM(t.TOTAL_VALUE) AS total_sales,
               COUNT(t.TRANSACTION_ID) AS total_transactions,
               COUNT(DISTINCT t.STORE_ID) AS unique_stores,
               AVG(t.TOTAL_VALUE) AS avg_sale_value
        FROM DIANA_SALES_ES.STOREFRONTS.SALESPERSONNEL s
        JOIN DIANA_SALES_ES.SALES.TRANSACTIONS t ON s.SALESREP_ID = t.SALESREP_ID
        JOIN DIANA_SALES_ES.STOREFRONTS.STORES sp ON sp.REGION_ID = s.REGION_ID
        GROUP BY s.SALESREP_ID, s.SALESREP_NAME, sp.REGION_ID
        """
        cur.execute(query)
        sales_data = cur.fetchall()

        logging.info(f"Fetched {len(sales_data)} rows of sales data.")
        
        cur.close()
        conn.close()

        return sales_data
    except Exception as e:
        logging.error(f"Error fetching data from Snowflake: {e}")
        return []

# Step 2: Enhanced Vectorization
def vectorize_sales_data(sales_data):
    logging.info("Initializing Pinecone and vectorizing sales data...")

    # Initialize Pinecone
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY", "your-pinecone-api-key"))
    
    index_name = "sales-personnel-performance"
    dimension = 4  # Adjust according to the number of features
    
    # Create or connect to Pinecone index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
        logging.info(f"Created Pinecone index: {index_name}")

    index = pinecone.Index(index_name)

    # Prepare vectorized sales data
    vectorized_sales = []
    for row in tqdm(sales_data, desc="Vectorizing sales data", unit="record"):
        salesrep_id, salesrep_name, region_id, total_sales, total_transactions, unique_stores, avg_sale_value = row
        vector = np.array([total_sales, total_transactions, unique_stores, avg_sale_value])
        
        # Prepare the vector object
        vectorized_sales.append({
            'id': str(salesrep_id),
            'values': vector.tolist(),
            'metadata': {
                'salesrep_name': salesrep_name,
                'region_id': region_id
            }
        })

    return index, vectorized_sales

# Step 3: Batch Upload to Pinecone with Progress Bar and Error Handling
def batch_upload_to_pinecone(index, vectorized_sales, batch_size=100):
    logging.info(f"Starting batch upload to Pinecone with batch size of {batch_size}...")

    try:
        for i in tqdm(range(0, len(vectorized_sales), batch_size), desc="Uploading to Pinecone", unit="batch"):
            batch = vectorized_sales[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logging.info("Data successfully uploaded to Pinecone.")
    except Exception as e:
        logging.error(f"Error during Pinecone upsert: {e}")

if __name__ == "__main__":
    # Fetch sales data
    sales_data = fetch_sales_data()

    # Check if any data was fetched
    if sales_data:
        # Vectorize the sales data and initialize the Pinecone index
        index, vectorized_sales = vectorize_sales_data(sales_data)

        # Upload vectorized data to Pinecone in batches
        batch_upload_to_pinecone(index, vectorized_sales, batch_size=100)
    else:
        logging.warning("No sales data fetched. Exiting...")
