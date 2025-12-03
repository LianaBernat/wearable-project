"""
run_all.py — Executes the full Capture-24 pipeline, including:

1. Chunked participant processing
2. Join chunk files per participant
3. Join all participants into final dataset
4. Train model on the final dataset
5. ....
"""

import time

from wearable.dataset_preparation.participants_pipeline import process_all_participants
from wearable.dataset_preparation.join_chunks_per_participant import join_chunks
from wearable.dataset_preparation.join_participants import join_participants


def main():

    print("\n===========================================")
    print(" STEP 1 — Processing all participants (chunked) ")
    print("===========================================\n")
    start = time.time()
    process_all_participants()
    print(f"\n[OK] Step 1 done in {(time.time() - start)/60:.2f} minutes.\n")

    print("\n===========================================")
    print(" STEP 2 — Joining chunk files per participant ")
    print("===========================================\n")
    start = time.time()
    join_chunks()
    print(f"\n[OK] Step 2 done in {(time.time() - start)/60:.2f} minutes.\n")

    print("\n===========================================")
    print(" STEP 3 — Joining all participants into final dataset ")
    print("===========================================\n")
    start = time.time()
    output = join_participants()
    print(f"\n[OK] Final dataset created: {output}")
    print(f"Total time: {(time.time() - start)/60:.2f} minutes.\n")


if __name__ == "__main__":
    main()
