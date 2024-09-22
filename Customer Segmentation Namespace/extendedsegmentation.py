import os
import logging
import numpy as np
from decimal import Decimal
import pinecone
import snowflake.connector
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

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

# Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "customer-classification"
embedding_dimension = 1536

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Adjusted to the free tier region
        )
    )

index = pc.Index(index_name)

# Fetch Segment Data from Snowflake
def fetch_segment_data():
    logging.info("Fetching segment data from Snowflake.")
    try:
        conn = snowflake.connector.connect(
            user=snowflake_user,
            password=snowflake_password,
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse='DIANA_DATA_LAKE',
            database='DIANA_SALES_ES',
            schema='SEGMENTS'
        )
        cur = conn.cursor()
        query = """
        SELECT sc.CRITERIA_ID, sc.SUBSEGMENT_ID, sc.VAC_MIN, sc.VAC_MAX, sc.FC_MIN, sc.FC_MAX, 
               sc.AC_MIN, sc.AC_MAX, sc.VMC_MIN, sc.VMC_MAX, sc.RUC_MAX, 
               sub.SUBSEGMENT_NAME, seg.SEGMENT_NAME
        FROM SEGMENTCRITERIA sc
        JOIN SUBSEGMENTS sub ON sc.SUBSEGMENT_ID = sub.SUBSEGMENT_ID
        JOIN SEGMENTS seg ON sub.SEGMENT_ID = seg.SEGMENT_ID
        """
        cur.execute(query)
        segment_data = cur.fetchall()
        return segment_data
    finally:
        cur.close()
        conn.close()

# Convert Decimal values to float safely
def safe_convert(x):
    if isinstance(x, Decimal):
        return float(x)
    return 0.0 if x is None else float(x)

# Vectorize the Data
def vectorize_segment_data(segment_data):
    logging.info("Vectorizing segment data.")
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_segment_row, segment_data), total=len(segment_data)))
    return results

def process_segment_row(row):
    (criteria_id, subsegment_id, vac_min, vac_max, fc_min, fc_max, ac_min, ac_max,
     vmc_min, vmc_max, ruc_max, subsegment_name, segment_name) = row

    # Apply safe conversion to vector values
    vector_values = [safe_convert(x) for x in [vac_min, vac_max, fc_min, fc_max, ac_min, ac_max, vmc_min, vmc_max, ruc_max]]
    
    # Ensure vector is properly defined here
    vector = np.zeros(embedding_dimension)
    vector[:len(vector_values)] = vector_values

    # Flatten criteria into separate metadata fields
    metadata = {
        'subsegment_name': str(subsegment_name),  # Ensure strings
        'segment_name': str(segment_name),  # Ensure strings
        'vac_min': float(safe_convert(vac_min)),
        'vac_max': float(safe_convert(vac_max)),
        'fc_min': float(safe_convert(fc_min)),
        'fc_max': float(safe_convert(fc_max)),
        'ac_min': float(safe_convert(ac_min)),
        'ac_max': float(safe_convert(ac_max)),
        'vmc_min': float(safe_convert(vmc_min)),
        'vmc_max': float(safe_convert(vmc_max)),
        'ruc_max': float(safe_convert(ruc_max))
    }
    
    return {'id': str(criteria_id), 'values': vector.tolist(), 'metadata': metadata}

# Upsert Vectors to Pinecone
def batch_upload_to_pinecone(vectors):
    logging.info(f"Uploading {len(vectors)} vectors to Pinecone.")
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)

# Main execution function
def main():
    segment_data = fetch_segment_data()
    if segment_data:
        vectorized_data = vectorize_segment_data(segment_data)
        batch_upload_to_pinecone(vectorized_data)
    logging.info("Data processing complete.")

if __name__ == "__main__":
    main()
