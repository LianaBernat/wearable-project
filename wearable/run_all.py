"""
run_all.py — Executes the full Capture-24 pipeline, including:

1. Chunked participant processing
2. Join chunk files per participant
3. Join all participants into final dataset
4. Train Random Forest model
5. Train MLP model
"""

import time

from wearable.dataset_preparation.participants_pipeline import process_all_participants
from wearable.dataset_preparation.join_chunks_per_participant import join_chunks
from wearable.dataset_preparation.join_participants import join_participants

from wearable.modeling.Random_forest.main import train_random_forest
from wearable.modeling.MLP.train import train_mlp


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
    print(f"Step 3 time: {(time.time() - start)/60:.2f} minutes.\n")

    print("\n===========================================")
    print(" STEP 4 — Training Random Forest model ")
    print("===========================================\n")
    start = time.time()
    rf_model = train_random_forest()
    print(f"\n[OK] Random Forest trained in {(time.time() - start)/60:.2f} minutes.\n")

    print("\n===========================================")
    print(" STEP 5 — Training MLP model ")
    print("===========================================\n")
    start = time.time()
    mlp_model, mlp_preprocessor = train_mlp()
    print(f"\n[OK] MLP trained in {(time.time() - start)/60:.2f} minutes.\n")

    print("\n===========================================")
    print(" PIPELINE FINISHED SUCCESSFULLY ")
    print("===========================================\n")


if __name__ == "__main__":
    main()
