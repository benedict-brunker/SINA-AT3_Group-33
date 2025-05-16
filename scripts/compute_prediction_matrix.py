# compute_prediction_matrix 

# --- Separate file needed for computing prediction matrix due to OOM errors --- 

# == Dependencies == 
import os 
import numpy as np 
import pandas as pd 
from scipy import sparse
from tqdm import tqdm 

# === Paths === 
project_dir = os.path.dirname(os.getcwd())
lookups_dir = os.path.join(project_dir, "lookups") 

# === Reload data === 

# Indices 
product_index = np.load(os.path.join(lookups_dir, "product_index.npy"), allow_pickle=True)
user_index = np.load(os.path.join(lookups_dir, "user_columns.npy"), allow_pickle=True) 
print("Indices loaded") 

# Matrices
U_sparse = sparse.load_npz(os.path.join(lookups_dir, "utility_matrix.npz"))
print("Utility matrix loaded")

S_sparse = sparse.load_npz(os.path.join(lookups_dir, "similarity_matrix.npz")) 
print("Similarity matrix loaded") 

# Reassemble dataframes
U = pd.DataFrame.sparse.from_spmatrix(
    U_sparse,
    index=product_index,
    columns=user_index
)

S = pd.DataFrame.sparse.from_spmatrix(
    S_sparse, 
    index=user_index, 
    columns=user_index 
)

print("Matrices assembled into dataframes")

# == Helper Function == 
def keep_top_k(sim_row, k): 
    top_k = sim_row.nlargest(k) 
    return sim_row.where(sim_row.index.isin(top_k.index), 0) 

# == Main Function == 
def predict_ratings(utility_matrix, similarity_matrix, k_only=True, k=5): 

    if k_only: 
        tqdm.pandas(desc="Filtering top-k similarities") 
        S_topk = similarity_matrix.progress_apply(keep_top_k, axis=1) 
    else: 
        S_topk = similarity_matrix.copy() 
    
    S_norm = S_topk.div(S_topk.sum(axis=0), axis=1)

    # downcast matrices to float32 to save memory 
    utility_matrix = utility_matrix.astype(np.float32) 
    S_norm = S_norm.astype(np.float32) 

    # compute prediction matrix as dot product of user similarity and known ratings
    predicted = utility_matrix.dot(S_norm) 

    # use the utility matrix to mark the known ratings (where no prediction is wanted)
    ## known ratings will be marked with nan and no recommendation with 0
    try:
        rated_mask = utility_matrix > 0
        predicted = predicted.mask(rated_mask, -1)
        return predicted, True
    except Exception as e: 
        print(f"{e}: saving prediction matrix before restoring original ratings.") 
        return predicted, False

# === Call === 
if __name__ == "__main__":

    print("Computing prediction matrix ...") 
    P, return_val = predict_ratings(U, S, k_only=False) 
    if return_val:
        print("Prediction matrix computed with original ratings restored.") 
        P_sparse = sparse.csr_matrix(P.values) 
        sparse.save_npz(os.path.join(lookups_dir, "prediction_matrix.npz"), P_sparse) 
        print(f"Matrix saved to {lookups_dir}") 
    else: 
        print("Prediction matrix computed with original ratings potentially overwritten.")
        P_sparse = sparse.csr_matrix(P.values) 
        sparse.save_npz(os.path.join(lookups_dir, "prediction_matrix_overwritten.npz"), P_sparse) 
        print(f"Matrix saved to {lookups_dir}")


