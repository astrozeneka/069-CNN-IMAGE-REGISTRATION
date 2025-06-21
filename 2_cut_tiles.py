import numpy as np
import tifffile
import cv2
import torch
import os


def generate_random_affine_params():
    """Generate random but reasonable affine transformation parameters"""
    # Small random variations around identity transformation
    scale_range = 0.1  # ±10% scaling
    rotation_range = 0.1  # ±~5.7 degrees in radians
    translation_range = 10  # ±10 pixels

    # Random scale and rotation
    scale_x = 1.0 + np.random.uniform(-scale_range, scale_range)
    scale_y = 1.0 + np.random.uniform(-scale_range, scale_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)

    # Translation
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)

    # Create affine matrix parameters [a, b, tx, c, d, ty]
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    a = scale_x * cos_r
    b = -scale_x * sin_r
    c = scale_y * sin_r
    d = scale_y * cos_r

    return np.array([a, b, tx, c, d, ty], dtype=np.float32)


def apply_affine_transform(image, affine_params):
    """Apply affine transformation to image using cv2"""
    # Convert parameters to 2x3 transformation matrix
    a, b, tx, c, d, ty = affine_params
    transform_matrix = np.array([[a, b, tx],
                                 [c, d, ty]], dtype=np.float32)

    # Apply transformation
    h, w = image.shape
    transformed = cv2.warpAffine(image, transform_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return transformed


def extract_random_tiles(image, tile_size=128, num_tiles=4):
    """Extract random tiles from image"""
    h, w = image.shape
    tiles = []
    positions = []

    for i in range(num_tiles):
        # Random top-left corner ensuring tile fits within image
        max_y = h - tile_size
        max_x = w - tile_size

        if max_y <= 0 or max_x <= 0:
            raise ValueError(f"Image too small for {tile_size}x{tile_size} tiles")

        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)

        # Extract tile
        tile = image[y:y + tile_size, x:x + tile_size]
        tiles.append(tile)
        positions.append((y, x))

    return tiles, positions


if __name__ == '__main__':
    # Parameters
    input_path = "single-channel.ome.tif"
    output_dir = "training_data"
    tile_size = 128
    num_tiles = 4

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the single-channel image
    print("Loading image...")
    image = tifffile.imread(input_path)

    # Ensure image is 2D
    if image.ndim > 2:
        image = image.squeeze()

    print(f"Image shape: {image.shape}")

    # Extract random tiles
    print(f"Extracting {num_tiles} random {tile_size}x{tile_size} tiles...")
    original_tiles, positions = extract_random_tiles(image, tile_size, num_tiles)

    # Generate transformations and create transformed tiles
    transformed_tiles = []
    affine_params_list = []

    for i, tile in enumerate(original_tiles):
        # Generate random affine parameters
        affine_params = generate_random_affine_params()
        affine_params_list.append(affine_params)

        # Apply transformation
        transformed_tile = apply_affine_transform(tile, affine_params)
        transformed_tiles.append(transformed_tile)

        # Save individual tiles
        tifffile.imwrite(f"{output_dir}/original_tile_{i:02d}.tif", tile)
        tifffile.imwrite(f"{output_dir}/transformed_tile_{i:02d}.tif", transformed_tile)

        print(f"Tile {i}: position ({positions[i][0]}, {positions[i][1]}), "
              f"affine params: {affine_params}")

    # Save all affine parameters
    affine_params_array = np.array(affine_params_list)
    np.save(f"{output_dir}/affine_parameters.npy", affine_params_array)

    # Save positions for reference
    positions_array = np.array(positions)
    np.save(f"{output_dir}/tile_positions.npy", positions_array)

    print(f"\nSaved {num_tiles} tile pairs to '{output_dir}/'")
    print(f"Affine parameters shape: {affine_params_array.shape}")
    print("Files created:")
    print("- original_tile_XX.tif (reference images)")
    print("- transformed_tile_XX.tif (transformed images)")
    print("- affine_parameters.npy (transformation parameters)")
    print("- tile_positions.npy (tile extraction positions)")