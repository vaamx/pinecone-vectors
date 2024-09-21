# Sparse vector example
sparse_vectors = {
    "id": product_id,
    "values": [(index, value) for index, value in enumerate(sparse_vector) if value != 0],
    "metadata": {"price": price}
}

index.upsert(vectors=[sparse_vectors])
