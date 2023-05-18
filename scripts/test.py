import bz2
import json

def convert_json_to_bz2(json_file, bz2_file):
    # Read the JSON file
    with open(json_file, 'r', encoding="utf-8") as f:
        data = f.read()

    # Compress the JSON data using bz2 compression
    compressed_data = bz2.compress(data.encode('utf-8'))

    # Write the compressed data to the .bz2 file
    with open(bz2_file, 'wb') as f:
        f.write(compressed_data)

jsonPath = "./data/raw/kaikii_fr.json"
bz2Path = "./data/raw/kaikii_fr.bz2"

convert_json_to_bz2(jsonPath,bz2Path)