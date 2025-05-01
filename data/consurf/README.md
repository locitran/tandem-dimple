# Folder tree:
```bash
.
├── consurf
│   ├── alignments
│   ├── data
│   ├── map_idx
│   ├── 2024-10-08.json
│   ├── main.py
|   ├── run.py
│   ├── README.md
```

# Description:
- `alignments` : Contains `.aln` files for a PDB chain, and it is the global alignment.
Example: `6Y4PB-6Y4OB.aln` file
```bash
----SKKAVWHKLLSKQRKRAVVACFRMA- # <-- PDB chain sequence
----|||||||||||||||||||||||||- 
SNARSKKAVWHKLLSKQRKRAVVACFRMAP # <-- ConsurfDB sequence
```
- `data` : Contains the ConSurfDB data retrieved from ConSurfDB.
- `map_idx` : Contains pickle `.pkl` file that saves a dictionary of the mapping of the PDB chain sequence to the ConsurfDB sequence. 
Example: `6Y4PB-6Y4OB.pkl` file
```bash
{
  'target': [-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1],
  'query': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
}
```
- `2024-10-08.json` : a lookup table to retrieve the unique chain ID corresponding to the PDB ID.
```bash
{
    "8RFN": {}, # No chain ID available
    "6E3I": {
        "A": "5UUKA" # PDB chain ID: 6E3IA, unique chain: 5UUKA
    },
...
}
```
- `main.py` and `run.py` : Contains the main functions to parse the ConSurfDB data and the main function to run the ConSurf parser.


I have added several files since it does not include in the ConSurfDB repository.
"1PCU" --> "1PCUA"
"1PFA" --> "1PFAA"
These files was created by ConSurfTool since 2024 when I started to work on ConSurf features (/mnt/nas_1/YangLab/loci/improve/consurf/tool_old). However, exactly configuration of the ConSurfTool is not available.

5XEZ: Not enough unique HMMER hits to run ConSurfTool
