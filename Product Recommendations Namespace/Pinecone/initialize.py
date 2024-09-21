import pinecone

# Initialize Pinecone
pinecone.init(api_key="edbfd83a-056b-4ffd-91d9-1d83a6a9c291", environment="us-east1-gcp")

# Check if the index exists, create if not
if "product-recommendations" not in pinecone.list_indexes():
    pinecone.create_index(
        name="product-recommendations",
        dimension=your_vector_dimension,
        metric="cosine"  # or another appropriate metric
    )
