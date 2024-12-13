from src.main import Tandem
import datetime

SAV_coords = ['Q92736 261 H R', 'Q92736 507 V I']

start_time = datetime.datetime.now()
job = Tandem(SAV_coords, job_name='Dec-09')
end_time = datetime.datetime.now()

print(f"Time taken: {(end_time - start_time).seconds:.2f} s or {(end_time - start_time).seconds / 60:.2f} min!")
