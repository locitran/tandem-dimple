import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import run
# from src.main import run

sav_list = ["P29033 170 N K"]

sav_list = ["A4D126 53 A T", "A4D2B0 114 H N", "A4D2B0 79 P H", "A8MYU2 768 W R", "A8MYU2 916 N S", "A9YTQ3 57 F S",
            "B2RXH2 113 Q R", "B2RXH2 258 F S", "B2RXH2 26 A T", "B2RXH2 42 Q R",
            "O00141 219 V I", "O00141 342 A V",  "O00170 16 R H", "O00180 268 C Y", "O00182 5 G S", "O00187 118 R C", 
            "O00187 120 D G", "O00187 128 T M", "O00187 155 H R", "O00187 371 D Y", "O00187 377 V A", "O00187 405 V M", 
            "O00187 439 R H", "O00187 99 R Q", "O00189 271 R H", "O00194 138 P L", "O00194 92 A T", "O00204 240 V I", 
            "O00204 51 L S", "O00206 175 T A", "O00206 188 Q R", "O00206 246 C S", "O00206 287 E D", "O00206 287 E G", 
            "O00206 306 C W", "O00206 310 V G", "O00206 329 N S", "O00206 342 F Y", "O00206 385 L F", "O00206 400 S N", 
            "O00206 443 F L", "O00206 46 Y C", "O00206 474 E K", "O00206 510 Q H", "O00206 73 S R", "O00212 144 R Q", 
            "O00214 184 R S", "O00214 19 F Y", "O00214 36 R C", "O00214 56 M V","O00217 102 R H","O00217 79 P L",
            "O00217 94 R C","O00222 265 I T","O00222 343 R Q","O00222 362 F Y","O00222 368 G D","O00222 392 R Q",
            "O00222 430 L F","O00232 358 V A","O00238 200 I N","O00238 224 R H","O00238 371 R Q", "O00255 110 G E", 
            "O00255 116 E G", "O00255 12 P L", "O00255 135 K I", "O00255 135 K M", "O00255 139 H D", "O00255 139 H P", 
            "O00255 139 H R", "O00255 139 H Y", "O00255 144 F V", "O00255 147 I F", "O00255 148 T P", "O00255 157 L W", 
            "O00255 158 D V", "O00255 158 D Y", "O00255 159 S I", "O00255 160 S F", "O00255 161 G C", "O00255 161 G D", 
            "O00255 161 G S", "O00255 165 A P", "O00255 165 A T", "O00255 167 V F", "O00255 169 A D", "O00255 170 C R", 
            "O00255 173 L P", "O00255 176 R Q", "O00255 177 D Y", "O00255 181 A P", "O00255 184 E D", "O00255 184 E K", 
            "O00255 184 E Q", "O00255 186 H R", "O00255 188 W R", "O00255 188 W S", "O00255 198 T I", "O00255 200 E G", 
            "O00255 22 L R", "O00255 220 V L"]

td = run(
    query=sav_list, # List of SAVs to be analyzed
    job_name='test/test_100SAVs', # Define where the job will be saved
    refresh=True, # Set to True to refresh the calculation
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
