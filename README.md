<<<<<<< HEAD

# Obtain MATLAB license for a computer from this (link)[https://www.mathworks.com/licensecenter/licenses/40638624/4628136/activations]
Step 1: Access to License Center
Step 2: Tag Install and Activate
Step 3: View activated computers
Step 4: Activated a computer
Step 5: Fill the form


You can then run TANDEM-DIMPLE using the following command:

```bash
docker run -it \
  --volume $HOME/input:/root/input \
  --volume $HOME/output:/root/output \
  --volume $HOME/matlab_license:/root/matlab_license \
  --mac-address=<MAC_ADDRESS_OF_MACHINE> \
  -e MLM_LICENSE_FILE=/root/matlab_license/license.lic \
  tandem \
  python main.py
```

<!-- In case testing and developing -->
```bash
docker run -it \
  --volume /mnt/Tsunami_HHD/newloci/NativeEnsembleWeb_copy/tandem:/tandem \
  --mac-address=58:11:22:00:83:bc \
  -e MLM_LICENSE_FILE=/tandem/license.lic \
  --name tandem \
  -u root \
  tandem bash
``` 

```bibtex
@article{Loci2025,
  author  = {Loci Tran, Chen-Hua Lu, Pei-Lung Chen, Lee-Wei Yang},
  journal = {Bioarchiv},
  title   = {Predicting the pathogenicity of SAVs Transfer-leArNing-ready and Dynamics-Empowered Model for DIsease-specific Missense Pathogenicity Level Estimation},
  year    = {2025},
  volume  = {*.*},
  number  = {*.*},
  pages   = {*.*},
  doi     = {*.*}
}
```   
=======
# tandem-dimple
Transfer-leArNing-ready and Dynamics-Empowered Model for DIsease-specific Missense Pathogenicity Level Estimation
>>>>>>> 931a28ebac40e06574efb8c08e582c39debb7d8e
