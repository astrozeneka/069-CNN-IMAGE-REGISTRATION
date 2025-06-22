import numpy as np
import tifffile
import cv2
import torch
import os


def generate_random_affine_params():
    """Generate random but reasonable affine transformation parameters"""
    # Small random variations around identity transformation
    scale_range = 0.01  # ±1% scaling
    rotation_range = 0.03  # ±~0.03 radians (about ±1.7 degrees)
    translation_range = 25  # ±10 pixels (might be changed later)

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

def apply_brightness_variation(image, brightness_range=0.2):
    """ Apply random brightness variation to the image """
    # Calculate random factor
    brightness_factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
    # Apply brightness adjustment
    adjusted_image = image * brightness_factor
    # Clip to a valid range
    adjusted_image = np.clip(adjusted_image, 0, image.max() if image.dtype != np.float32 else 1.0)
    # Return
    return adjusted_image.astype(image.dtype)

def is_background_tile(tile, intensity_threshold=0.1, min_std=4000):
    """
    Check if a tile is mostly background (too dark) for immunohistochemistry images

    Args:
        tile: 2D numpy array representing the image tile
        intensity_threshold: minimum mean intensity (normalized 0-1)
        min_std: minimum standard deviation to ensure some structure

    Returns:
        bool: True if tile is background (should be skipped), False otherwise
    """
    # Normalize tile to 0-1 range
    tile_norm = tile.astype(np.float32)
    if tile_norm.max() > 1:
        tile_norm = tile_norm / tile_norm.max()

    # Calculate mean intensity
    mean_intensity = np.mean(tile_norm)

    # Calculate standard deviation (low std indicates uniform background)
    # Note that "tile" is used instead of "tile_norm" to detect uniformity
    std_intensity = np.std(tile)

    # Print information
    print(f"Tile mean intensity: {mean_intensity:.3f}")

    # Check if tile is too dark or too uniform
    is_too_dark = mean_intensity < intensity_threshold
    is_too_uniform = std_intensity < min_std

    #if std_intensity < 0.1:
    #    print()
    #    # save tile as a tiff
    #    #tifffile.imwrite("background_tile.tif", tile_norm)
    #else:
    #    print()


    print("==================")
    #print(f"Tile is too dark: {is_too_dark}, mean intensity: {mean_intensity:.3f}")
    print(f"Tile is too uniform: {is_too_uniform}, std intensity: {std_intensity:.3f}")

    return is_too_uniform


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


def extract_random_tiles(image, tile_size=128, num_tiles=4, max_attempts=100):
    """
    Extract random tiles from image, skipping background tiles

    Args:
        image: 2D numpy array
        tile_size: size of square tiles to extract
        num_tiles: number of tiles to extract
        max_attempts: maximum attempts to find valid tiles

    Returns:
        tiles: list of valid (non-background) tiles
        positions: list of (y, x) positions for each tile
    """
    h, w = image.shape
    tiles = []
    positions = []
    attempts = 0

    if h < tile_size or w < tile_size:
        raise ValueError(f"Image too small for {tile_size}x{tile_size} tiles")

    max_y = h - tile_size
    max_x = w - tile_size

    while len(tiles) < num_tiles and attempts < max_attempts:
        # Random top-left corner
        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)

        # Extract tile
        tile = image[y:y + tile_size, x:x + tile_size]

        # Check if tile is background
        print("==> Extracting tile no. ", len(tiles))
        if not is_background_tile(tile):
            tiles.append(tile)
            positions.append((y, x))
            print(f"Valid tile {len(tiles)}: position ({y}, {x}), "
                  f"mean intensity: {np.mean(tile):.3f}")
        else:
            print(f"Skipping background tile at ({y}, {x}), "
                  f"mean intensity: {np.mean(tile):.3f}")

        attempts += 1

    if len(tiles) < num_tiles:
        print(f"Warning: Only found {len(tiles)} valid tiles out of {num_tiles} requested "
              f"after {max_attempts} attempts")

    return tiles, positions


def extract_transformed_tile_from_parent(parent_image, original_pos, affine_params, tile_size):
    """
    Extract a tile from parent image at transformed coordinates

    Args:
        parent_image: The full parent image
        original_pos: (y, x) position of the original tile
        affine_params: affine transformation parameters [a, b, tx, c, d, ty]
        tile_size: size of the tile

    Returns:
        transformed_tile: tile extracted from transformed coordinates
    """
    y_orig, x_orig = original_pos
    a, b, tx, c, d, ty = affine_params

    # Calculate center of original tile
    center_x = x_orig + tile_size // 2
    center_y = y_orig + tile_size // 2

    # Apply affine transformation to center coordinates
    new_center_x = a * center_x + b * center_y + tx
    new_center_y = c * center_x + d * center_y + ty

    # Calculate new top-left corner
    new_x = int(new_center_x - tile_size // 2)
    new_y = int(new_center_y - tile_size // 2)

    # Ensure coordinates are within image bounds
    h, w = parent_image.shape
    new_x = max(0, min(new_x, w - tile_size))
    new_y = max(0, min(new_y, h - tile_size))

    # Extract tile from parent image at new coordinates
    transformed_tile = parent_image[new_y:new_y + tile_size, new_x:new_x + tile_size]

    return transformed_tile


import argparse
parser = argparse.ArgumentParser(description="Extract and transform image tiles")
# num_tiles
parser.add_argument('--num_tiles', type=int, default=40,
                    help='Number of tiles to extract from the image')
parser.add_argument('--tile_size', type=int, default=128,
                    help='Size of each tile to extract (in pixels)')
parser.add_argument('--input_path', type=str, default="single-channel.ome.tif",
                    help='Path to the input image file')
parser.add_argument('--output_dir', type=str, default="training_data",
                    help='Directory to save the extracted tiles and parameters')
parser.add_argument('--brightness_range', type=float, default=0.2,
                    help='Range of brightness variation (0.1 = ±10%)')
args = parser.parse_args()

if __name__ == '__main__':
    # Parameters
    input_path = args.input_path
    output_dir = args.output_dir
    tile_size = args.tile_size
    num_tiles = args.num_tiles

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
        transformed_tile = extract_transformed_tile_from_parent(
            image, positions[i], affine_params, tile_size
        )
        transformed_tiles.append(transformed_tile)

        # Generate the brightness variation to both original and transformed tiles
        tile = apply_brightness_variation(tile, args.brightness_range)
        transformed_tile = apply_brightness_variation(transformed_tile, args.brightness_range)

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