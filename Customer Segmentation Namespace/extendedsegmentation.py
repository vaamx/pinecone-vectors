import os
import logging
import numpy as np
from decimal import Decimal
from pinecone import Pinecone, ServerlessSpec
import snowflake.connector
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv
import openai
import math

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Set up API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
snowflake_user = os.getenv("SNOWFLAKE_USER")
snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API key
openai.api_key = openai_api_key

# Pinecone Initialization
index_name = "diana-sales"
embedding_dimension = 1536
namespace = "segmentcriteria"

# Initialize Pinecone using the new API method
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust region as necessary
    )
index = pc.Index(index_name)

# Function to generate embeddings using OpenAI API
def generate_embeddings(text):
    """
    Generates embeddings for given text using OpenAI's embedding model.
    """
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"  # Choose a suitable embedding model
    )
    embedding = response['data'][0]['embedding']  # Assuming the response structure
    return embedding

# Function to safely convert decimals to floats
def safe_convert(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0  # or some default value
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0  # Default value in case of conversion error

# Fetch Segment Data from Snowflake
def fetch_segment_data():
    logging.info("Fetching segment data from Snowflake.")
    try:
        conn = snowflake.connector.connect(
            user=snowflake_user,
            password=snowflake_password,
            account=snowflake_account,
            warehouse='DIANA_DATA_LAKE',
            database='DIANA_SALES_ES',
            schema='SEGMENTS'
        )
        query = """
        SELECT seg.SEGMENT_ID, seg.SEGMENT_NAME, sub.SUBSEGMENT_ID, sub.SUBSEGMENT_NAME,
               cri.CRITERIA_ID, cri.VAC_MIN, cri.VAC_MAX, cri.FC_MIN, cri.FC_MAX,
               cri.AC_MIN, cri.AC_MAX, cri.VMC_MIN, cri.VMC_MAX, cri.RUC_MAX, cri.IL_DESCRIPTION
        FROM DIANA_SALES_ES.SEGMENTS.SEGMENTS seg
        LEFT JOIN DIANA_SALES_ES.SEGMENTS.SUBSEGMENTS sub ON seg.SEGMENT_ID = sub.SEGMENT_ID
        LEFT JOIN DIANA_SALES_ES.SEGMENTS.SEGMENTCRITERIA cri ON sub.SUBSEGMENT_ID = cri.SUBSEGMENT_ID
        ORDER BY seg.SEGMENT_ID, sub.SUBSEGMENT_ID, cri.CRITERIA_ID;
        """
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        logging.error(f"Error fetching data from Snowflake: {e}")
        return []
    finally:
        conn.close()

def process_segment_row(row):
    try:
        # Unpack the row values
        (segment_id, segment_name, subsegment_id, subsegment_name, criteria_id, 
         vac_min, vac_max, fc_min, fc_max, ac_min, ac_max, vmc_min, vmc_max, ruc_max, 
         il_description) = row

        # Safely convert numerical values
        numerical_values = [safe_convert(vac_min), safe_convert(vac_max), safe_convert(fc_min), safe_convert(fc_max),
                            safe_convert(ac_min), safe_convert(ac_max), safe_convert(vmc_min), safe_convert(vmc_max),
                            safe_convert(ruc_max)]

        # Generate text embedding for example purposes (replace with your embedding model)
        text_to_embed = f"{segment_name} {subsegment_name}"
        embedding = generate_embeddings(text_to_embed)  # Use your embedding model here

        # Check the available space in the vector for embedding
        available_slots = 1536 - len(numerical_values)
        if len(embedding) > available_slots:
            embedding = embedding[:available_slots]  # Truncate the embedding if it's too large

        # Ensure the vector has real data
        vector = np.zeros(1536)  # Assuming a 1536-dimensional vector
        vector[:len(numerical_values)] = numerical_values
        vector[len(numerical_values):len(numerical_values) + len(embedding)] = embedding

        # Construct metadata
        metadata = {
            'segment_name': segment_name,
            'subsegment_name': subsegment_name,
            'criteria_id': criteria_id,
            'vac_min': safe_convert(vac_min),
            'vac_max': safe_convert(vac_max),
            'fc_min': safe_convert(fc_min),
            'fc_max': safe_convert(fc_max),
            'ac_min': safe_convert(ac_min),
            'ac_max': safe_convert(ac_max),
            'vmc_min': safe_convert(vmc_min),
            'vmc_max': safe_convert(vmc_max),
            'ruc_max': safe_convert(ruc_max),
            'il_description': il_description
        }

        return {'id': str(criteria_id), 'values': vector.tolist(), 'metadata': metadata}

    except Exception as e:
        logging.error(f"Error processing row {row}: {e}")
        return None



# Vectorize the Segment Data using ThreadPoolExecutor for concurrency
def vectorize_segment_data(segment_data):
    logging.info("Vectorizing segment data.")
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_segment_row, segment_data), total=len(segment_data)))
    return results

# Batch Upload the Vectorized Data to Pinecone
def batch_upload_to_pinecone(vectors, batch_size=100):
    logging.info(f"Uploading {len(vectors)} vectors to Pinecone in batches of {batch_size}.")
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone", unit="batch"):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            logging.info(f"Uploaded batch {i // batch_size + 1}")
        except Exception as e:
            logging.error(f"Error during Pinecone upsert: {e}")

# Main Execution Function
def main():
    # Step 1: Fetch segment data from Snowflake
    segment_data = fetch_segment_data()
    
    if not segment_data:
        logging.error("No segment data fetched. Exiting.")
        return
    
    # Step 2: Vectorize the segment data
    vectorized_data = vectorize_segment_data(segment_data)
    
    # Step 3: Batch upload the vectors to Pinecone
    batch_upload_to_pinecone(vectorized_data)

if __name__ == "__main__":
    main()
