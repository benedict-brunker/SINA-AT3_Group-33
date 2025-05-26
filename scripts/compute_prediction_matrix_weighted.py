# compute_prediction_matrix_weighted 

# this script computes by weighted average of user similarity 
# there is no k constraint as dissimilar users are expected to be zeroed out 

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
user_index = np.load(os.path.join(lookups_dir, "user_index.npy"), allow_pickle=True) 
print("Indices loaded") 

# Matrices
U_sparse = sparse.load_npz(os.path.join(lookups_dir, "utility_matrix.npz"))
print("Utility matrix loaded")

S_sparse = sparse.load_npz(os.path.join(lookups_dir, "user_similarity_matrix.npz")) 
print("Similarity matrix loaded") 


# == Main Function == 
def predict_ratings(utility_matrix, similarity_matrix): 

    # transposed to items as cols for faster slicing 
    utility_csc = utility_matrix.T.tocsc() 
    print("Shape of utility_csc:", utility_csc.shape)
    n_users, n_items = utility_csc.shape 
    print("Num users in utility_csc:", n_users) 
    print("Num items in utility_csc:", n_items)
    
    predicted = sparse.lil_matrix((n_users, n_items), dtype=np.float32) 

    # iterate through users 
    for x in range(n_users): 
        # get user's similarity vector 
        sim_x = similarity_matrix[x, :].toarray().ravel() 

        # iterate through items 
        for i in range(n_items): 
            # continue if actual rating exists 
            if utility_matrix[x, i] != 0: 
                predicted[x, i] = -1  # placeholder for actual rating available
                continue 
            # get users who rated item i
            col = utility_csc.getcol(i) 
            y_idx = col.indices 
            y_ratings = col.data 

            sims = sim_x[y_idx] 

            if sims.sum() == 0: 
                predicted[x, i] = np.nan 
                continue 

            numerator = np.dot(sims, y_ratings) 
            denominator = np.sum(np.abs(sims)) 
            if denominator != 0: 
                predicted[x, i] = numerator / denominator
            
            if x == 0 and i == 0: 
                print("Prediction for first item generated for first user")

        if x == 0: 
            print("Predictions generated for first user") 

    return predicted 

# === Call === 
if __name__ == "__main__":

    print("Computing prediction matrix ...") 
    P_lil = predict_ratings(U_sparse, S_sparse) 
    P_sparse = P_lil.tocsr() 
    sparse.save_npz(os.path.join(lookups_dir, "prediction_matrix.npz")) 
    print(f"Matrix saved to {lookups_dir}")



    # if return_val:
    #     print("Prediction matrix computed with original ratings restored.") 
    #     P_sparse = sparse.csr_matrix(P.values) 
    #     sparse.save_npz(os.path.join(lookups_dir, "prediction_matrix.npz"), P_sparse) 
    #     print(f"Matrix saved to {lookups_dir}") 
    # else: 
    #     print("Prediction matrix computed with original ratings potentially overwritten.")
    #     P_sparse = sparse.csr_matrix(P.values) 
    #     sparse.save_npz(os.path.join(lookups_dir, "prediction_matrix_overwritten.npz"), P_sparse) 
    #     print(f"Matrix saved to {lookups_dir}")