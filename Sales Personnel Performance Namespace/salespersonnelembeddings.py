import os
import logging
from tqdm import tqdm
import numpy as np
import openai
import pinecone
import snowflake.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Step 0: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load sensitive keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Step 1: Fetch Sales Data from Snowflake
def fetch_sales_data():
    logging.info("Connecting to Snowflake...")
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA')
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

# Step 2: Vectorize Descriptions using OpenAI Embeddings
def vectorize_description(description):
    try:
        # OpenAI API call for embeddings
        response = openai.Embedding.create(
            input=description,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        logging.error(f"Error in OpenAI embedding call: {e}")
        return None

# Step 3: Vectorize Sales Data and Use OpenAI API for Descriptions
def vectorize_sales_data(sales_data, embedding_dimension=1536, namespace="default"):
    logging.info("Initializing Pinecone and vectorizing sales data...")

    # Initialize Pinecone
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "diana-sales"
    index = pinecone.Index(index_name)

    # Prepare vectorized sales data
    vectorized_sales = []

    def process_row(row):
        salesrep_id, salesrep_name, region_id, total_sales, total_transactions, unique_stores, avg_sale_value = row
        # Create a 1536-dimensional vector, where first four dimensions contain sales data
        vector = np.zeros(embedding_dimension)
        vector[:4] = np.array([total_sales, total_transactions, unique_stores, avg_sale_value])

        metadata = {
            'salesrep_name': salesrep_name,
            'region_id': region_id
        }

        return {'id': str(salesrep_id), 'values': vector.tolist(), 'metadata': metadata}

    # Process sales data in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row) for row in sales_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vectorizing Sales Data", unit="record"):
            try:
                result = future.result()
                if result:
                    vectorized_sales.append(result)
            except Exception as e:
                logging.error(f"Error in vectorization: {e}")

    logging.info(f"Vectorized {len(vectorized_sales)} sales records.")
    return index, vectorized_sales, namespace

# Step 4: Batch Upload to Pinecone with Progress Bar and Error Handling
def batch_upload_to_pinecone(index, vectorized_sales, namespace, batch_size=100):
    logging.info(f"Starting batch upload to Pinecone with batch size of {batch_size}...")

    try:
        for i in tqdm(range(0, len(vectorized_sales), batch_size), desc="Uploading to Pinecone", unit="batch"):
            batch = vectorized_sales[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            logging.info(f"Uploaded batch {i // batch_size + 1} to Pinecone.")
        
        logging.info("Data successfully uploaded to Pinecone.")
    except Exception as e:
        logging.error(f"Error during Pinecone upsert: {e}")

# Main Function: Execute the Workflow
def main(namespace="salespersonnel-data"):
    sales_data = fetch_sales_data()

    if sales_data:
        # Vectorize the sales data and initialize the Pinecone index
        index, vectorized_sales, namespace = vectorize_sales_data(sales_data, embedding_dimension=1536, namespace=namespace)

        # Upload vectorized data to Pinecone in batches
        batch_upload_to_pinecone(index, vectorized_sales, namespace, batch_size=100)
    else:
        logging.warning("No sales data fetched. Exiting...")

if __name__ == "__main__":
    main(namespace="salespersonnel-data")
