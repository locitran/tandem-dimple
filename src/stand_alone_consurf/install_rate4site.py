
import os
import shutil
import subprocess

def submit_job_to_Q(job_name_prefix, cmd):

    process = subprocess.Popen(cmd, shell=True)
    process.communicate()

def install_rate4site(rate4site_dir, rate4site_slow_dir):

	
    # create directory for rate4site
    submit_job_to_Q("download_rate4site", "git clone https://github.com/barakav/r4s_for_collab.git")

    # create directory for rate4site slow
    shutil.copytree(rate4site_dir, rate4site_slow_dir)

    # make rate4site
    submit_job_to_Q("install_rate4site", "cd %s\nmake\nchmod 755 rate4site" %rate4site_dir)
    
    # change the make file 
    os.remove(rate4site_slow_dir + "Makefile") 
    os.rename(rate4site_slow_dir + "Makefile_slow", rate4site_slow_dir + "Makefile") 

    # make rate4site
    submit_job_to_Q("install_rate4site", "cd %s\nmake\nchmod 755 rate4site" %rate4site_slow_dir)
    
current_dir = os.getcwd()
install_rate4site(current_dir + "/r4s_for_collab/", current_dir + "/r4s_for_collab_slow/")