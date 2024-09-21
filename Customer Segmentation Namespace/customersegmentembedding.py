import snowflake.connector
import numpy as np
import pinecone
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

logging.info("Step 1: Fetch Data from Snowflake")

# Step 1: Fetch Data from Snowflake
try:
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user='OPSCALEAI',
        password='Opscale2030',
        account='nvvmnod-mw08757',
        warehouse='DIANA_DATA_LAKE',
        database='DIANA_SALES_ES',
        schema='SALES'
    )
    cur = conn.cursor()

    # SQL query to aggregate store purchase data and join with demographic information
    query = """
    SELECT st.STORE_ID, st.STORE_NAME,
           SUM(t.TOTAL_VALUE) AS total_sales,
           COUNT(t.TRANSACTION_ID) AS total_transactions,
           AVG(t.TOTAL_VALUE) AS avg_transaction_value,
           COUNT(DISTINCT t.PURCHASE_DATE) AS purchase_frequency,
           pd.POPULATION_DENSITY, pd.TOTAL_POPULATION, pd.URBAN_POPULATION, pd.AREA_SQ_KM,
           r.DEPARTMENT, r.MUNICIPALITY
    FROM DIANA_SALES_ES.SALES.TRANSACTIONS t
    JOIN DIANA_SALES_ES.STOREFRONTS.STORES st ON t.STORE_ID = st.STORE_ID
    JOIN DIANA_SALES_ES.STOREFRONTS.POPULATIONDENSITY pd ON st.REGION_ID = pd.DENSITY_ID
    JOIN DIANA_SALES_ES.STOREFRONTS.REGION r ON st.REGION_ID = r.REGION_ID
    GROUP BY st.STORE_ID, st.STORE_NAME, pd.POPULATION_DENSITY, pd.TOTAL_POPULATION, pd.URBAN_POPULATION, pd.AREA_SQ_KM, r.DEPARTMENT, r.MUNICIPALITY
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

# Step 2: Vectorization with Additional Features (Aligned with 1536-dimensional embedding)
logging.info("Step 2: Vectorizing Data")

# Initialize Pinecone
pinecone.init(api_key="edbfd83a-056b-4ffd-91d9-1d83a6a9c291")

# Connect to the existing Pinecone index
index_name = "diana-sales"
index = pinecone.Index(index_name)
logging.info(f"Connected to Pinecone index: {index_name}")

# Vectorize and prepare data for Pinecone (in a specific namespace)
namespace = "store-segmentation"  # Define your namespace

vectorized_stores = []
for row in tqdm(store_data, desc="Vectorizing stores"):
    (store_id, store_name, total_sales, total_transactions, avg_transaction_value,
     purchase_frequency, pop_density, total_population, urban_population, area_sq_km,
     department, municipality) = row

    # Placeholder: Extend or update the feature vector as needed to align with the 1536 dimensions
    vector = np.zeros(1536)  # OpenAI Embedding Model: Ensure consistency with 1536 dimensions
    vector[:7] = [
        total_sales,
        total_transactions,
        avg_transaction_value,
        purchase_frequency,
        pop_density,
        total_population,
        urban_population
    ]

    # Add metadata for the store
    vectorized_stores.append({
        'id': str(store_id),
        'values': vector.tolist(),
        'metadata': {
            'store_name': store_name,
            'department': department,
            'municipality': municipality,
            'area_sq_km': area_sq_km
        }
    })

logging.info(f"Vectorized data for {len(vectorized_stores)} stores")

# Step 3: Batch Upload to Pinecone with Enhanced Vectors and Namespace
logging.info("Step 3: Batch Upload to Pinecone")

# Define batch size for uploading to Pinecone
batch_size = 100
for i in tqdm(range(0, len(vectorized_stores), batch_size), desc="Uploading batches to Pinecone"):
    batch = vectorized_stores[i:i + batch_size]
    try:
        index.upsert(vectors=batch, namespace=namespace)
        logging.info(f"Uploaded batch {i // batch_size + 1} to namespace '{namespace}'")
    except Exception as e:
        logging.error(f"Error uploading batch {i // batch_size + 1} to Pinecone: {e}")

logging.info("All batches uploaded successfully")
