import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        'filename', help='filename of dataset to query metadata for'
    )
    args = p.parse_args()

    df_metadata = pd.read_json('metadata.json')
    query = df_metadata[df_metadata['file'] == args.filename]
    print(query)

if __name__ == '__main__':
    main()
