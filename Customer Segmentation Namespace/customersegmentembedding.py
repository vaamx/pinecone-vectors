import os
import logging
import numpy as np
import pinecone
import snowflake.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# Set up API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
snowflake_user = os.getenv("SNOWFLAKE_USER")
snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API key
openai.api_key = openai_api_key

# Pinecone Initialization
index_name = "diana-sales"
embedding_dimension = 1536
namespace = "store-segmentation"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
    )
index = pc.Index(index_name)

# Step 1: Fetch Data from Snowflake
def fetch_sales_data():
    try:
        conn = snowflake.connector.connect(
            user=snowflake_user,
            password=snowflake_password,
            account='nvvmnod-mw08757',
            warehouse='DIANA_DATA_LAKE',
            database='DIANA_SALES_ES',
            schema='SALES'
        )
        cur = conn.cursor()
        query = """
        SELECT st.STORE_ID, st.STORE_NAME,
            SUM(t.TOTAL_VALUE) AS total_sales,
            COUNT(t.TRANSACTION_ID) AS total_transactions,
            AVG(t.TOTAL_VALUE) AS avg_transaction_value,
            COUNT(DISTINCT t.PURCHASE_DATE) AS purchase_frequency,
            pd.POPULATION_DENSITY, pd.TOTAL_POPULATION, pd.AREA_SQ_KM, -- Removed pd.URBAN_POPULATION
            r.DEPARTMENT, r.MUNICIPALITY
        FROM DIANA_SALES_ES.SALES.TRANSACTIONS t
        JOIN DIANA_SALES_ES.STOREFRONTS.STORES st ON t.STORE_ID = st.STORE_ID
        JOIN DIANA_SALES_ES.STOREFRONTS.POPULATIONDENSITY pd ON st.REGION_ID = pd.DENSITY_ID
        JOIN DIANA_SALES_ES.STOREFRONTS.REGION r ON st.REGION_ID = r.REGION_ID
        GROUP BY st.STORE_ID, st.STORE_NAME, pd.POPULATION_DENSITY, pd.TOTAL_POPULATION, pd.AREA_SQ_KM, r.DEPARTMENT, r.MUNICIPALITY
        """
        cur.execute(query)
        store_data = cur.fetchall()
        logging.info(f"Fetched {len(store_data)} rows from Snowflake")
    except Exception as e:
        logging.error(f"Error fetching data from Snowflake: {e}")
        store_data = []
    finally:
        cur.close()
        conn.close()
        logging.info("Snowflake connection closed")
    return store_data

# Step 2: Vectorization with Concurrency
def process_row(row):
    try:
        (store_id, store_name, total_sales, total_transactions, avg_transaction_value,
         purchase_frequency, pop_density, total_population, urban_population, area_sq_km,
         department, municipality) = row

        # Placeholder for creating a 1536-dimensional vector, using first 7 slots for our custom features
        vector = np.zeros(embedding_dimension)
        vector[:7] = np.array([
            total_sales,
            total_transactions,
            avg_transaction_value,
            purchase_frequency,
            pop_density,
            total_population,
            urban_population
        ])

        metadata = {
            'store_name': store_name,
            'department': department,
            'municipality': municipality,
            'area_sq_km': area_sq_km
        }

        return {'id': str(store_id), 'values': vector.tolist(), 'metadata': metadata}
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return None

# Vectorization with ThreadPoolExecutor
def vectorize_sales_data(store_data):
    vectorized_sales = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row) for row in store_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vectorizing Sales Data", unit="record"):
            result = future.result()
            if result:
                vectorized_sales.append(result)
    return vectorized_sales

# Step 3: Batch Upload to Pinecone
def batch_upload_to_pinecone(vectors, batch_size=100):
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone", unit="batch"):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            logging.info(f"Uploaded batch {i // batch_size + 1}")
        except Exception as e:
            logging.error(f"Error during Pinecone upsert: {e}")

# Main execution function
def main():
    # Step 1: Fetch sales data from Snowflake
    store_data = fetch_sales_data()

    # Step 2: Vectorize the sales data
    vectorized_sales = vectorize_sales_data(store_data)

    # Step 3: Batch upload the vectors to Pinecone
    batch_upload_to_pinecone(vectorized_sales)

if __name__ == "__main__":
    main()
