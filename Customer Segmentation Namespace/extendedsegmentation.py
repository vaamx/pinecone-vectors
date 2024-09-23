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
index_name = "segmentcriteria"
namespace = "segmentcriteria"  # Ensure namespace is clearly defined
embedding_dimension = 1536

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(name=index_name)

# Verify or create the namespace
if namespace not in index.list_namespaces():
    index.create_namespace(name=namespace)

# Fetch Segment Data from Snowflake
def fetch_segment_data():
    logging.info("Fetching segment data from Snowflake.")
    with snowflake.connector.connect(
        user=snowflake_user,
        password=snowflake_password,
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse='DIANA_DATA_LAKE',
        database='DIANA_SALES_ES',
        schema='SEGMENTS'
    ) as conn:
        with conn.cursor() as cur:
            query = """
            SELECT sc.CRITERIA_ID, sc.SUBSEGMENT_ID, sc.VAC_MIN, sc.VAC_MAX, sc.FC_MIN, sc.FC_MAX,
                   sc.AC_MIN, sc.AC_MAX, sc.VMC_MIN, sc.VMC_MAX, sc.RUC_MAX,
                   sub.SUBSEGMENT_NAME, seg.SEGMENT_NAME
            FROM SEGMENTCRITERIA sc
            JOIN SUBSEGMENTS sub ON sc.SUBSEGMENT_ID = sub.SUBSEGMENT_ID
            JOIN SEGMENTS seg ON sub.SEGMENT_ID = seg.SEGMENT_ID
            """
            cur.execute(query)
            return cur.fetchall()

# Convert Decimal values to float safely
def safe_convert(x):
    return float(x) if isinstance(x, Decimal) else 0.0 if x is None else float(x)

# Vectorize the Data
def vectorize_segment_data(segment_data):
    logging.info("Vectorizing segment data.")
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_segment_row, segment_data), total=len(segment_data)))
    return results

def process_segment_row(row):
    criteria_id, subsegment_id, *values, subsegment_name, segment_name = row
    vector_values = [safe_convert(x) for x in values]
    vector = np.zeros(embedding_dimension)
    vector[:len(vector_values)] = vector_values
    metadata = {
        'subsegment_name': subsegment_name,
        'segment_name': segment_name,
        'criteria': {str(idx): val for idx, val in enumerate(values)}
    }
    return {'id': str(criteria_id), 'values': vector.tolist(), 'metadata': metadata}

# Upsert Vectors to Pinecone
def batch_upload_to_pinecone(vectors):
    logging.info(f"Uploading {len(vectors)} vectors to Pinecone in namespace '{namespace}'.")
    response = index.upsert(vectors=vectors, namespace=namespace)
    logging.info(f"Upsert response: {response}")

# Main execution function
def main():
    segment_data = fetch_segment_data()
    if segment_data:
        vectorized_data = vectorize_segment_data(segment_data)
       
