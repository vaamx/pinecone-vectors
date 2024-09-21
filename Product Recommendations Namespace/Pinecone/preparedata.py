import some_vectorization_library

# Example function to vectorize text
def vectorize_text(text):
    return some_vectorization_library.encode(text)

# Prepare data
product_vectors = [(str(product_id), vectorize_text(description), {"price": price}) for product_id, description, price in your_product_data]
