def vectorize_data(data):
    # Assuming you have a vectorization function
    return [vectorize_text(description) for _, description, _ in data]

# Vectorize descriptions
vectorized_data = vectorize_data(rows)
