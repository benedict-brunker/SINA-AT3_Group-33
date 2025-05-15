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
U_sparse = sparse.load_npz(os.path.join(lookups_dir, "utility_matrix.npz"))
U_rows = np.load(os.path.join(lookups_dir, "U_rows.npy"), allow_pickle=True)
U_cols = np.load(os.path.join(lookups_dir, "U_cols.npy"), allow_pickle=True) 
print("Utility matrix loaded")

S_sparse = sparse.load_npz(os.path.join(lookups_dir, "similarity_matrix.npz")) 
S_rows = np.load(os.path.join(lookups_dir, "S_rows.npy"), allow_pickle=True) 
S_cols = np.load(os.path.join(lookups_dir, "U_cols.npy"), allow_pickle=True) 
print("Similarity matrix loaded") 

# Reassemble dataframes
U = pd.DataFrame.sparse.from_spmatrix(
    U_sparse,
    index=U_rows,
    columns=U_cols
)
S = pd.DataFrame.sparse.from_spmatrix(
    S_sparse, 
    index=S_rows, 
    columns=S_cols 
)
"Matrices assembled into dataframes"

# == Helper Function == 
def keep_top_k(sim_row, k): 
    top_k = sim_row.nlargest(k) 
    return sim_row.where(sim_row.index.isin(top_k.index), 0) 

# == Main Function == 
def predict_ratings(utility_matrix, similarity_matrix, k_only=True, k=5): 
    utility_filled = utility_matrix.fillna(0) 
    if k_only: 
        tqdm.pandas(desc="Filtering top-k similarities") 
        S_topk = similarity_matrix.progress_apply(keep_top_k, axis=1) 
    else: 
        S_topk = similarity_matrix.copy() 
    
    S_topk = S_topk.fillna(0) 
    S_norm = S_topk.div(S_topk.sum(axis=0), axis=1).fillna(0) 

    predicted = utility_filled.dot(S_norm) 

    predicted[~utility_matrix.isna()] = utility_matrix[~utility_matrix.isna()] 

    return predicted 

# === Call === 
if __name__ == "__main__":

    print("Computing prediction matrix ...") 
    P = predict_ratings(U, S, k_only=False) 
    print("Prediction matrix computed.") 
    # log 
    P_sparse = sparse.csr_matrix(P.values) 
    P_rows = P.index.tolist() 
    P_cols = P.index.tolist() 
    sparse.save_npz(os.path.join(lookups_dir, "prediction_matrix.npz"), P_sparse) 
    np.save(os.path.join(lookups_dir, "P_rows.npy"), P_rows) 
    np.save(os.path.join(lookups_dir, "P_cols.npy"), P_cols) 
    print(f"Matrix saved to {lookups_dir}") 



