import faiss
import numpy as np
import time

# Generate some random data for demonstration
d = 128  # Dimensionality of vectors
nb = 10000  # Number of database vectors
nq = 5  # Number of query vectors

np.random.seed(42)  # For reproducibility
data = np.random.random((nb, d)).astype('float32')
queries = np.random.random((nq, d)).astype('float32')

# Helper function to create and train an index
def create_index(index_type, d, data, nlist=100, nprobe=10, m=8):
    """
    index_type: str, type of index ('Flat', 'IVF', 'PQ', 'HNSW')
    d: int, dimensionality of vectors
    data: np.ndarray, training data
    nlist: int, number of clusters for IVF
    nprobe: int, number of clusters to search for IVF
    m: int, number of sub-quantizers for PQ
    """
    if index_type == 'Flat':
        index = faiss.IndexFlatL2(d)  # Exact search
    elif index_type == 'IVF':
        quantizer = faiss.IndexFlatL2(d)  # Used for clustering
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(data)  # Train the clustering
        index.nprobe = nprobe  # Number of clusters to search
    elif index_type == 'PQ':
        quantizer = faiss.IndexFlatL2(d)  # Used for clustering
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        index.train(data)  # Train the product quantizer
        index.nprobe = nprobe
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(d, m)  # HNSW graph with m neighbors
        index.hnsw.efSearch = 32  # Number of candidates in search
        index.hnsw.efConstruction = 40  # Number of candidates in construction
    else:
        raise ValueError("Unsupported index type. Choose from: Flat, IVF, PQ, HNSW.")
    
    index.add(data)  # Add data to the index
    return index

# Test all index types and perform comparison
index_types = ['Flat', 'IVF', 'PQ', 'HNSW']

print(f"{'='*50}")
print("Performance Comparison")
print(f"{'='*50}")

for index_type in index_types:
    try:
        print(f"\n{'-'*30}")
        print(f"Testing {index_type} Index")
        print(f"{'-'*30}")
        
        # Measure index creation time
        start_time = time.time()
        index = create_index(index_type, d, data)
        creation_time = time.time() - start_time
        
        # Perform a search with query vectors
        k = 5  # Number of nearest neighbors
        start_time = time.time()
        distances, indices = index.search(queries, k)
        search_time = time.time() - start_time
        
        # Output the results
        print(f"Index type: {index_type}")
        print(f"Number of vectors indexed: {index.ntotal}")
        print(f"Creation time: {creation_time:.4f}s")
        print(f"Search time: {search_time:.6f}s")
        print("Distances:\n", distances)
        print("Indices:\n", indices)
        
    except Exception as e:
        print(f"Error with {index_type} index: {e}")