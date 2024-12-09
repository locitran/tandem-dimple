from pathlib import Path
from .pipeline import Pipeline
from .utils.logger import Logger
from .utils.settings import ROOT_DIR
from .predict.modules import ModelInference

SAV_col_width = 20
PDB_col_width = 20
model_col_width = 15

class Tandem:

    def __init__(
            self,
            # input_file, # Contains the SAV coordinates
            SAV_coords,
            job_name="Tandem",
            output_dir='.',
            custom_pdb=None,
            models=None,
            model_names=None,
            ):

        if model_names is None and models is None:
            model_names = [f'TANDEM_{i}' for i in range(1, 6)]
        
        self.SAV_coords = SAV_coords
        self.job_name = job_name

        self.output_dir = Path(output_dir) / job_name
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(exist_ok=True)
        
        tandem_logger = Logger(self.output_dir)
        tandem_logger.info(f"Starting log info Tandem job: {self.job_name}")
        tandem_logger.warning(f"Starting log warning Tandem job: {self.job_name}")
        tandem_logger.error(f"Starting log error Tandem job: {self.job_name}")

        p = self.run()
        featMatrix = p._get_features()
        featExtention = p._get_extension()

        mi = ModelInference(models)
        preds = mi(featMatrix)

        tandem_logger.info(f"Finished Tandem job: {self.job_name}")

        # Save predictions as text file
        # First column: SAV coordinates
        # Second column: PDB coordinates
        # Third column: Predictions of the model 1
        # Fourth column: Predictions of the model 2
        # ...
        # Last column: Predictions of the model 5
        output_file = self.output_dir / 'predictions.txt'
        with open(output_file, 'w') as f:
            for i, sav in enumerate(SAV_coords):
                SAV_coord = sav
                PDB_coord = featExtention['PDB_coords'][i]
                pred = preds[i]
                
                # Header
                f.write(
                    f"{'Uniprot/pos/wt/mt':<{SAV_col_width}}{'PDB/chain/pos/wt':<{PDB_col_width}}"
                    f"{''.join(f'{name:<{model_col_width}}' for name in model_names)}\n"
                )

                # Data
                f.write(
                    f"{SAV_coord:<{SAV_col_width}}{PDB_coord:<{PDB_col_width}}"
                    f"{''.join(f'{x:<{model_col_width}.3f}' for x in pred)}\n"
                )


    def run(self):
        p = Pipeline(SAV_coords=self.SAV_coords, output_dir=self.output_dir)
        return p

if __name__ == '__main__':
    SAV_coords = ['Q92736 261 H R']
    job = Tandem(SAV_coords, job_name='Nov-13')


