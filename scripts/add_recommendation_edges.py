# add_recommendation_edges 

# --- Separate file needed to avoid memory errors --- 

# === Dependencies === 
import os 
import networkx as nx 
import numpy as np
from scipy import sparse
import pickle

# === Directories === 
project_dir = os.path.dirname(os.getcwd()) 
lookups_dir = os.path.join(project_dir, "lookups") 
graphs_dir = os.path.join(project_dir, "graphs")

# === Load === 
def load(dirs=[lookups_dir, graphs_dir], paths=["prediction_matrix.npz", "user_columns.npy", "product_index.npy", "graph.pkl"]): 
    # matrix 
    P_sparse = sparse.load_npz(os.path.join(dirs[0], paths[0]))
    P_coo = P_sparse.tocoo() 
    del P_sparse
    user_cols = np.load(os.path.join(dirs[0], paths[1]), allow_pickle=True) 
    product_index = np.load(os.path.join(dirs[0], paths[2]), allow_pickle=True)
    # graph 
    with open(os.path.join(dirs[1], paths[3]), 'rb') as f: 
        G = pickle.load(f)
    
    
if __name__ == "__main__":

    # Reload graph 

    # Reload prediction matrix 
    
    
    # Add edges 
