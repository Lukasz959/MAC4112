import argparse
import time
from src.calculate_Features import extract_features
from src.calculate_Features import extract_signal


def main():
    parser = argparse.ArgumentParser(description="Extract signal features")
    parser.add_argument("--folder", default="data")
    parser.add_argument("--trial")
    parser.add_argument("--key")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    # Start clock
    start_time = time.perf_counter()
    # Extract sensor signals
    signals = extract_signal(args.folder, args.trial, args.key)
    # Extract features from signals
    df = extract_features(signals)
    # Write features to a .csv file if requested
    if args.write:
        write_features(df)
    print(df.head(5))
    # Display elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()