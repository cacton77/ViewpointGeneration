import time
import numpy as np
import cupy as cp
import open3d as o3d


def estimate_curvature(pcd, nn_glob):
    # Estimate normals and curvature of the set point cloud
    print('Estimating curvature...')

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    curvature = np.empty((len(points), 1), dtype=np.float32)

    # Check if the normals are already present
    if normals.size == 0:
        print('\tNormals are not present. Estimating normals...')
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)

    # Estimate curvature
    for i in range(len(points)):
        # Access the points in the vicinity of the current point
        nn_loc = points[nn_glob[i]]
        # Calculate the covariance matrix of the points in the vicinity
        COV = np.cov(nn_loc, rowvar=False)
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigval, eigvec = np.linalg.eig(COV)
        # Sort the eigenvalues in ascending order
        idx = np.argsort(eigval)
        # Store the curvature of the point
        curvature[i] = eigval[idx][0] / np.sum(eigval)

    return curvature


def estimate_curvature_cupy(pcd, nn_glob):
    """
    Parallelized version using CuPy for GPU acceleration
    """
    # Estimate curvature of the set point cloud
    print('Estimating curvature with GPU...')

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Check if the normals are already present
    if normals.size == 0:
        print('\tNormals are not present. Estimating normals...')
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)

    # Transfer data to GPU
    start_time = time.time()
    points_gpu = cp.asarray(points)
    nn_glob_gpu = cp.asarray(nn_glob)
    end_time = time.time()
    print(
        f'Transferring data to GPU took {end_time - start_time:.4f} seconds')

    # Get neighborhood sizes for vectorization
    max_neighbors = max(len(neighbors) for neighbors in nn_glob)
    n_points = len(points)

    # Create padded neighbor array for efficient GPU processing
    # Shape: (n_points, max_neighbors, 3)
    start_time = time.time()
    neighbor_coords = cp.zeros((n_points, max_neighbors, 3), dtype=cp.float32)
    neighbor_masks = cp.zeros((n_points, max_neighbors), dtype=cp.bool_)
    end_time = time.time()
    print(
        f'Creating padded neighbor array took {end_time - start_time:.4f} seconds')

    # Fill neighbor coordinates (this part could be optimized further)
    start_time = time.time()
    for i in range(n_points):
        neighbors = nn_glob[i]
        n_neighbors = len(neighbors)
        neighbor_coords[i, :n_neighbors] = points_gpu[neighbors]
        neighbor_masks[i, :n_neighbors] = True
    end_time = time.time()
    print(
        f'Filling neighbor coordinates took {end_time - start_time:.4f} seconds')

    # Vectorized curvature computation
    # Time vectorized computation
    start_time = time.time()
    curvature_gpu = compute_curvature_vectorized(
        neighbor_coords, neighbor_masks)
    end_time = time.time()
    print(
        f'Vectorized curvature computation took {end_time - start_time:.4f} seconds')

    # Transfer result back to CPU
    curvature = cp.asnumpy(curvature_gpu).reshape(-1, 1)

    return curvature


def compute_curvature_vectorized(neighbor_coords, neighbor_masks):
    """
    Vectorized curvature computation using CuPy

    Args:
        neighbor_coords: (n_points, max_neighbors, 3) - coordinates of neighbors
        neighbor_masks: (n_points, max_neighbors) - mask for valid neighbors

    Returns:
        curvature: (n_points,) - curvature values
    """
    n_points, max_neighbors, _ = neighbor_coords.shape

    # Compute means for each neighborhood
    # Shape: (n_points, 1, 3)
    neighbor_counts = cp.sum(neighbor_masks, axis=1, keepdims=True)
    means = cp.sum(neighbor_coords *
                   neighbor_masks[..., None], axis=1, keepdims=True) / neighbor_counts[..., None]

    # Center the coordinates
    # Shape: (n_points, max_neighbors, 3)
    centered_coords = neighbor_coords - means
    centered_coords = centered_coords * \
        neighbor_masks[..., None]  # Zero out invalid neighbors

    # Compute covariance matrices for all points simultaneously
    # Shape: (n_points, 3, 3)
    covariance_matrices = cp.matmul(
        centered_coords.transpose(0, 2, 1),  # (n_points, 3, max_neighbors)
        centered_coords  # (n_points, max_neighbors, 3)
    ) / (neighbor_counts[..., None] - 1)  # Unbiased estimator

    # Compute eigenvalues for all covariance matrices
    eigenvalues = cp.linalg.eigvalsh(
        covariance_matrices)  # Shape: (n_points, 3)

    # Sort eigenvalues and compute curvature
    # Sort along the eigenvalue dimension
    eigenvalues_sorted = cp.sort(eigenvalues, axis=1)
    curvature = eigenvalues_sorted[:, 0] / cp.sum(eigenvalues_sorted, axis=1)

    return curvature

# Method 1: Vectorized Index Assignment (Most Compatible)


def fill_neighbors_vectorized_v1(points_gpu, nn_glob):
    """
    Vectorized approach using advanced indexing
    """
    n_points = len(nn_glob)
    max_neighbors = max(len(neighbors) for neighbors in nn_glob)

    neighbor_coords = cp.zeros((n_points, max_neighbors, 3), dtype=cp.float32)
    neighbor_masks = cp.zeros((n_points, max_neighbors), dtype=cp.bool_)

    start_time = time.time()

    # Create flat arrays for vectorized assignment
    point_indices = []
    neighbor_indices = []
    coord_indices = []

    for i, neighbors in enumerate(nn_glob):
        n_neighbors = len(neighbors)
        point_indices.extend([i] * n_neighbors)
        neighbor_indices.extend(neighbors)
        coord_indices.extend(list(range(n_neighbors)))

    # Convert to CuPy arrays
    point_idx_gpu = cp.array(point_indices)
    neighbor_idx_gpu = cp.array(neighbor_indices)
    coord_idx_gpu = cp.array(coord_indices)

    # Vectorized assignment
    neighbor_coords[point_idx_gpu,
                    coord_idx_gpu] = points_gpu[neighbor_idx_gpu]
    neighbor_masks[point_idx_gpu, coord_idx_gpu] = True

    end_time = time.time()
    print(f'Vectorized v1 took {end_time - start_time:.4f} seconds')

    return neighbor_coords, neighbor_masks

# Method 2: Pre-allocate and Use CuPy Scatter


def fill_neighbors_vectorized_v2(points_gpu, nn_glob):
    """
    Using CuPy's advanced indexing with pre-computed indices
    """
    n_points = len(nn_glob)
    max_neighbors = max(len(neighbors) for neighbors in nn_glob)

    start_time = time.time()

    # Pre-compute all indices on CPU (faster than nested loops)
    neighbor_lengths = np.array([len(neighbors) for neighbors in nn_glob])
    total_neighbors = np.sum(neighbor_lengths)

    # Flatten everything
    flat_neighbors = np.concatenate(nn_glob)
    point_repeats = np.repeat(np.arange(n_points), neighbor_lengths)
    coord_positions = np.concatenate(
        [np.arange(length) for length in neighbor_lengths])

    # Transfer to GPU
    flat_neighbors_gpu = cp.array(flat_neighbors)
    point_repeats_gpu = cp.array(point_repeats)
    coord_positions_gpu = cp.array(coord_positions)

    # Initialize arrays
    neighbor_coords = cp.zeros((n_points, max_neighbors, 3), dtype=cp.float32)
    neighbor_masks = cp.zeros((n_points, max_neighbors), dtype=cp.bool_)

    # Vectorized assignment
    neighbor_coords[point_repeats_gpu,
                    coord_positions_gpu] = points_gpu[flat_neighbors_gpu]
    neighbor_masks[point_repeats_gpu, coord_positions_gpu] = True

    end_time = time.time()
    print(f'Vectorized v2 took {end_time - start_time:.4f} seconds')

    return neighbor_coords, neighbor_masks

# Method 3: Ragged Array Approach (Most Memory Efficient)


def fill_neighbors_ragged(points_gpu, nn_glob):
    """
    Use ragged arrays to avoid padding - more memory efficient
    """
    start_time = time.time()

    # Flatten all neighbor indices
    neighbor_lengths = [len(neighbors) for neighbors in nn_glob]
    flat_neighbors = cp.array(np.concatenate(nn_glob))

    # Get all neighbor coordinates at once
    # Shape: (total_neighbors, 3)
    all_neighbor_coords = points_gpu[flat_neighbors]

    # Create split indices to reconstruct per-point neighborhoods
    split_indices = cp.cumsum(cp.array(neighbor_lengths[:-1]))

    end_time = time.time()
    print(f'Ragged array approach took {end_time - start_time:.4f} seconds')

    return all_neighbor_coords, split_indices, neighbor_lengths

# Method 4: Custom CUDA Kernel (Maximum Performance)


def fill_neighbors_cuda_kernel(points_gpu, nn_glob):
    """
    Custom CUDA kernel for maximum performance
    """
    n_points = len(nn_glob)
    max_neighbors = max(len(neighbors) for neighbors in nn_glob)

    start_time = time.time()

    # Prepare data for kernel
    neighbor_lengths = np.array([len(neighbors)
                                for neighbors in nn_glob], dtype=np.int32)
    flat_neighbors = np.concatenate(nn_glob).astype(np.int32)
    neighbor_offsets = np.concatenate(
        [[0], np.cumsum(neighbor_lengths[:-1])]).astype(np.int32)

    # Transfer to GPU
    neighbor_lengths_gpu = cp.array(neighbor_lengths)
    flat_neighbors_gpu = cp.array(flat_neighbors)
    neighbor_offsets_gpu = cp.array(neighbor_offsets)

    neighbor_coords = cp.zeros((n_points, max_neighbors, 3), dtype=cp.float32)
    neighbor_masks = cp.zeros((n_points, max_neighbors), dtype=cp.bool_)

    # Custom kernel
    fill_kernel = cp.ElementwiseKernel(
        'int32 point_idx, raw int32 neighbor_lengths, raw int32 neighbor_offsets, raw int32 flat_neighbors, raw float32 points',
        'raw float32 neighbor_coords, raw bool neighbor_masks',
        '''
        int n_neighbors = neighbor_lengths[point_idx];
        int offset = neighbor_offsets[point_idx];
        int max_neighbors = neighbor_coords.shape()[1];
        
        for (int j = 0; j < n_neighbors; j++) {
            int neighbor_idx = flat_neighbors[offset + j];
            int coord_idx = point_idx * max_neighbors * 3 + j * 3;
            int mask_idx = point_idx * max_neighbors + j;
            
            neighbor_coords[coord_idx + 0] = points[neighbor_idx * 3 + 0];
            neighbor_coords[coord_idx + 1] = points[neighbor_idx * 3 + 1];
            neighbor_coords[coord_idx + 2] = points[neighbor_idx * 3 + 2];
            neighbor_masks[mask_idx] = true;
        }
        ''',
        'fill_neighbors_kernel'
    )

    # Execute kernel
    point_indices = cp.arange(n_points, dtype=cp.int32)
    fill_kernel(point_indices, neighbor_lengths_gpu, neighbor_offsets_gpu,
                flat_neighbors_gpu, points_gpu.ravel(),
                neighbor_coords, neighbor_masks)

    end_time = time.time()
    print(f'CUDA kernel took {end_time - start_time:.4f} seconds')

    return neighbor_coords, neighbor_masks

# Method 5: Hybrid Approach with Batching


def fill_neighbors_batched(points_gpu, nn_glob, batch_size=1000):
    """
    Process in batches to balance memory and speed
    """
    n_points = len(nn_glob)
    max_neighbors = max(len(neighbors) for neighbors in nn_glob)

    neighbor_coords = cp.zeros((n_points, max_neighbors, 3), dtype=cp.float32)
    neighbor_masks = cp.zeros((n_points, max_neighbors), dtype=cp.bool_)

    start_time = time.time()

    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_nn = nn_glob[batch_start:batch_end]

        # Use vectorized approach for this batch
        batch_coords, batch_masks = fill_neighbors_vectorized_v2(
            points_gpu, batch_nn)

        neighbor_coords[batch_start:batch_end] = batch_coords
        neighbor_masks[batch_start:batch_end] = batch_masks

    end_time = time.time()
    print(f'Batched approach took {end_time - start_time:.4f} seconds')

    return neighbor_coords, neighbor_masks

# Method 6: Most Practical Optimized Version


def fill_neighbors_optimized_practical(points_gpu, nn_glob):
    """
    Most practical optimization - good balance of speed and simplicity
    """
    n_points = len(nn_glob)
    max_neighbors = max(len(neighbors) for neighbors in nn_glob)

    start_time = time.time()

    # Pre-compute on CPU (often faster than GPU for small operations)
    neighbor_lengths = np.array([len(neighbors) for neighbors in nn_glob])
    cumsum_lengths = np.concatenate([[0], np.cumsum(neighbor_lengths)])

    # Create flat arrays
    flat_neighbors = np.concatenate(nn_glob)
    point_indices = np.repeat(np.arange(n_points), neighbor_lengths)
    local_indices = np.concatenate(
        [np.arange(length) for length in neighbor_lengths])

    # Single GPU transfer and vectorized operations
    flat_neighbors_gpu = cp.array(flat_neighbors)
    point_indices_gpu = cp.array(point_indices)
    local_indices_gpu = cp.array(local_indices)

    # Initialize output arrays
    neighbor_coords = cp.zeros((n_points, max_neighbors, 3), dtype=cp.float32)
    neighbor_masks = cp.zeros((n_points, max_neighbors), dtype=cp.bool_)

    # Single vectorized assignment
    neighbor_coords[point_indices_gpu,
                    local_indices_gpu] = points_gpu[flat_neighbors_gpu]
    neighbor_masks[point_indices_gpu, local_indices_gpu] = True

    end_time = time.time()
    print(f'Optimized practical took {end_time - start_time:.4f} seconds')

    return neighbor_coords, neighbor_masks

# Usage example and performance comparison


def compare_methods(points_gpu, nn_glob):
    """
    Compare all methods
    """
    print("Comparing neighbor filling methods:")
    print("=" * 50)

    methods = [
        ("Original (for reference)", None),  # Your original method
        ("Vectorized v1", fill_neighbors_vectorized_v1),
        ("Vectorized v2", fill_neighbors_vectorized_v2),
        ("Optimized Practical", fill_neighbors_optimized_practical),
        ("Ragged Array", fill_neighbors_ragged),
        ("Batched", fill_neighbors_batched),
    ]

    for name, method in methods:
        if method is None:
            continue
        try:
            result = method(points_gpu, nn_glob)
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")

    print("=" * 50)
    print("Recommendation: Use 'Optimized Practical' for best balance of speed and reliability")

# Example usage in your curvature function


def estimate_curvature_optimized(pcd, nn_glob):
    """
    Your curvature function with optimized neighbor filling
    """
    print('Estimating curvature with optimized method...')

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if normals.size == 0:
        print('Normals are not present. Estimating normals...')
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)

    # Transfer data to GPU
    points_gpu = cp.asarray(points)

    # Use optimized neighbor filling
    neighbor_coords, neighbor_masks = fill_neighbors_optimized_practical(
        points_gpu, nn_glob)

    # Continue with vectorized curvature computation...
    curvature_gpu = compute_curvature_vectorized(
        neighbor_coords, neighbor_masks)

    return cp.asnumpy(curvature_gpu).reshape(-1, 1)
