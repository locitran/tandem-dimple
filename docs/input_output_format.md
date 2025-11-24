# Input

## `query` (required)

We could query a SAV, a batch of SAVs, a UniPort ID, or UniProt ID with mutation site. A
SAV should be formatted as `UniPort ID`, `position`, `wt aa`, `mut aa` ("wt" and "mut" refer to wild-type and mutant), where `position` is the place that mutated amino acid (aa) is introduced.

*   One SAV: 
    ```python
    query = ['O14508 52 S N'] # 1 SAV
    ```
*   Batch of SAVs:
    ```python
    query = ['O14508 52 S N', 'A4D2B0 114 H N'] # 2 SAVs
    ```
*   UniProt ID:
    ```python
    query = 'P29033' # (protein length x 19) SAVs
    ```

*   UniProt ID with mutation site:
    ```python
    query = 'P29033 10' # 19 SAVs
    ```

## `job_name` (required)

This will be used to create a directory for storing intermediate and output files. Job folders are stored at [../jobs/](../jobs/) directory. If provided job name is not unique, the job will overwrite the previous one.

## `models` (optional)

Default is None, it means that we will use the general disease model, TANDEM-DIMPLE, stored at [models/different_number_of_layers/20250423-1234-tandem/n_hidden-5](../models/different_number_of_layers/20250423-1234-tandem/n_hidden-5).

Besides TANDEM-DIMPLE, we can also use transfer-learned models for specific diseases, such as GJB2 and RYR1.
*   GJB2:
    ```python
    models = '../models/transfer_learning_GJB2'
    ```
*   RYR1:
    ```python
    models = '../models/TransferLearning_RYR1'
    ```

## `custom_PDB` (optional)

If `custom_PDB` is provided, we will use the custom PDB structure to map the SAV and calculate structural dynamics features. `custom_PDB` could be a structure file in `.cif` or `.pdb` format, or a PDB ID that can be downloaded from the PDB database.
*   Custom PDB file:
    ```python
    custom_PDB = 'examples/AF-P29033-F1-model_v4.pdb'
    ```
*   PDB ID:
    ```python
    custom_PDB = '2ZW3'
    ```

If `custom_PDB` is from Alphafold2 database (in `.pdb` format), it should be started with `AF-`. 
If `custom_PDB` is a Alphafold3 predicted structure (in `.cif` format), it should contain `alphafoldserver.com/output-terms` in the first line of the file.

If `custom_PDB` is a structure file, ConSurf will compute the conservation score (slower); if it’s a PDB ID, the score is fetched from the ConSurf database (faster).


## `featSet` (optional)

The default is None, which means that we will use the latest feature set.
If you want to use a specific feature set, you can specify it as a list.

## `refresh` (optional)

The default is False, which means that we will use the cached data and precomputed features in the pickle files.
We have two files to cache, corresponding SEQ and STR/DYN features. 

*   SEQ features are stored in [../data/pickles/uniprot/](../data/pickles/uniprot/) directory.
*   STR/DYN features are stored in [../data/pickles/pdb/](../data/pickles/pdb/) directory.
If `refresh` is True, we will re-compute the features and overwrite the cached files.

# Output

## `log.txt`

Take a look in case of error. It contains the log of the program.
Example: [examples/log.txt](../examples/log.txt)

## `SAVs.txt`

This file contains the list of SAVs that are used in the job, separated by new lines.
It is generated from the input query.

## `job_name-Uniprot2PDB.txt`

This file contains the mapping SAVs to PDB structures.

| SAV_coords        | Unique_SAV_coords | Asymmetric_PDB_coords | BioUnit_PDB_coords    | OPM_PDB_coords     | Asymmetric_PDB_resolved_length |
|------------------|-------------------|------------------------|------------------------|--------------------|-------------------------------|
| P29033 52 V A     | P29033 52 V A      | 2ZW3 A 52 V            | 2ZW3 A 52 V 1          | 2ZW3 A 52 V        | 216                           |

*   `SAV_coords`: SAV coordinates in the input query.
*   `Unique_SAV_coords`: Unique SAV coordinates in the input query, in case input UniProt ID is obsolete.
*   `Asymmetric_PDB_coords`: Coordinates of Asymmetric Unit of the PDB structure.
*   `BioUnit_PDB_coords`: Coordinates of Biological Unit of the PDB structure.
*   `OPM_PDB_coords`: Coordinates of Available OPM structure for given PDB ID.
*   `Asymmetric_PDB_resolved_length`: Length of the resolved structure in the asymmetric unit.

## `job_name-features.csv`
This file contains the features of the SAVs in the job. The first column is SAV_coords, and the rest are features, 33 features supposedly, which then are the input of model inference.

## `job_name-report.txt`
Predicted results are stored in this file. 

| SAVs             | Probability | Decision   | Voting |
|------------------|-------------|------------|--------|
| P29033 52 V A     | 0.3384      | Benign     | 100.0  |

*   `SAVs`: SAV coordinates in the input query.
*   `Probability`: Pathogenicity probability of the SAV.
*   `Decision`: Pathogenic or Benign. 
*   `Voting`: Percentage of voting for the decision.

## `job_name-full_predictions.txt`
Detailed predictions of each model are stored in this file.

| SAVs             | TANDEM_0 | TANDEM_1 | TANDEM_2 | TANDEM_3 | TANDEM_4 |
|------------------|----------|----------|----------|----------|----------|
| P29033 52 V A     | 0.3451   | 0.3731   | 0.3031   | 0.3156   | 0.3551   |


