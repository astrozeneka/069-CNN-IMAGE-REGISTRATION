import tifffile

image_path="/Users/ryanrasoarahona/Desktop/CODEUR-PROJECT/Projects/051-nextflow-mcmicro/work/58/401d46da176a5f060992fc828b2c64/exemplar-001.ome.tif"

if __name__ == '__main__':
    # Read th multi-channel OME-TIFF file
    image = tifffile.imread(image_path)

    # Extract the first channel
    first_channel = image[0]

    # Save to a new TIFF file
    tifffile.imwrite('first_channel.tif', first_channel)

    print("Done")