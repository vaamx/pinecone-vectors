# Connect to your index
index = pinecone.Index("product-recommendations")

# Upload data
index.upsert(vectors=product_vectors)
