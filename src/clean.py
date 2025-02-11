import pandas as pd
import json
import csv
from tqdm import tqdm


def main():
    dataset_path = "data/yelp-2015.json"

    # Get the data
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)[["stars", "text"]]

    csv_filename = "data/yelp-2015.csv"
    df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()
