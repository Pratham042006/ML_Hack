"""Create a small synthetic experiment fixture. Not real data or results."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trip_id", "observation_index", "arc_id", "distance_m", "label"])
        for trip in range(6):
            for obs in range(8):
                for arc in range(3):
                    # Deliberately toy data, with one positive per observation.
                    positive = (obs + trip) % 3
                    distance = 2 + ((obs * 7 + trip * 3 + arc * 11) % 20)
                    if arc == positive:
                        distance *= 0.4
                    writer.writerow([f"synthetic_{trip}", obs, arc, distance, int(arc == positive)])
    print(f"Synthetic fixture only: {path}")


if __name__ == "__main__":
    main()
