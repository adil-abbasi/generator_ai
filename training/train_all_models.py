import pandas as pd

from config import TRAIN_PATH
from train_model import train_target


def main():
    print("=" * 70)
    print("Generator AI - Training All Target Models")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load training dataset
    # ---------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)

    # ---------------------------------------------------------
    # Find all target columns automatically
    # ---------------------------------------------------------

    target_columns = [
        column
        for column in train_df.columns
        if column.startswith("target_")
    ]

    print(f"\nFound {len(target_columns)} target columns.\n")

    for index, target in enumerate(target_columns, start=1):

        print("\n" + "=" * 70)
        print(f"[{index}/{len(target_columns)}] Training {target}")
        print("=" * 70)

        try:
            train_target(target)

            print(f"\n✓ Finished training {target}")

        except Exception as error:

            print(f"\n✗ Failed training {target}")

            print(error)

    print("\n" + "=" * 70)
    print("All Training Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()