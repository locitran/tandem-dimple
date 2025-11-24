import os
from multiprocessing import Pool, cpu_count
from datetime import datetime
import json
from main import get_all_pdb_ids, get_unique_chains, get_consurf

current_time = datetime.now().strftime('%Y-%m-%d')
db_dir = os.path.join('db', current_time)
os.makedirs(db_dir, exist_ok=True)

def process_pdb_id(pdb_id):
    # Function to process a single PDB ID
    db_entry = {}
    unique_chains = get_unique_chains(pdb_id)
    if unique_chains is None:
        return (pdb_id, '')

    for chain, unique_chain in unique_chains:
        df = get_consurf(unique_chain, db_dir)
        db_entry[chain] = unique_chain if df is not None else ''

    return (pdb_id, db_entry)

current_time = datetime.now().strftime('%Y-%m-%d')
db_dir = os.path.join('db', current_time)
os.makedirs(db_dir, exist_ok=True)

# Get all PDB IDs
all_ids = get_all_pdb_ids()

# Define the number of CPUs to use
num_cpus = min(8, cpu_count())

# Create a Pool for parallel processing
with Pool(processes=num_cpus) as pool:
    results = pool.map(process_pdb_id, all_ids)  # Process only the first 10 for now

# Build the database dictionary from the results
db = {pdb_id: db_entry for pdb_id, db_entry in results}

# Save the results to a JSON file
json_file = os.path.join('db', f'{current_time}.json')
with open(json_file, 'w') as f:
    json.dump(db, f, indent=4)

