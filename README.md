# CT_HeadAndNeck_OARs
This repository provides pre-trained deep learning models for segmenting the organs at risk in radiotherapy treatment planning
of head and neck cancer patients. It operates on axial CT scans acquired for this purpose in the head-first supine (HFS) orientation.  
              
### OAR group 1 
Segmented using DeepLabV3+ model ensembles [1] for sequential localization to delineate:  
* Left masseter    
* Right masseter    
* Left medial pterygoid    
* Right medial pterygoid    
* Larynx    
* Pharyngeal constrictor muscle     
    
### OAR group 2
Segmented using self-attention U-net network [2] to delineate:    
* Left parotid    
* Right parotid    
* Left submandibular gland  
* Right submandibular gland  
* Mandible    
* Spinal cord    
* Brain stem    
* Oral cavity      

## Usage

### Inputs
The following input options are supported: 
* DICOM (single CT): Input path to directory containing DICOM H&N CT images.
* DICOM (cohort)   : Input path to hierarchical directory with sub-directories per patient holding DICOM H&N CT images.
* NIfTI (single CT): Input path to NIfTI file containing H&N CT image.
* NIfTI (cohort)   : Input path to flat directory of NIfTI H&N CT images per patient. 

Input file format is inferred from the organization of data.

### Demo
See [Jupyter notebook](https://github.com/cerr/pyCERR-Notebooks/blob/main/autosegment_CT_HeadAndNeck_OARs.ipynb) for sample workflow. 

### Installation    
* Use the [model installer](https://github.com/cerr/model_installer/) to download the trained models. 

```
git clone https://github.com/cerr/model_installer.git
cd model_installer
./installer.sh -m 6 -p N
```

* Create a virtual environment and install dependencies (from [requirements.txt](https://github.com/cerr/CT_HeadAndNeck_OARs/blob/main/requirements.txt).
   
Using [uv](https://github.com/astral-sh/uv)
```
uv venv .venv 
source .venv/bin/activate
uv pip install --no-deps -r  requirements.txt --extra-index-url https://download.pytorch.org/whl/cu92 --index-strategy unsafe-best-match
```
        
Or using Conda:    
```
conda create -y --name CT_HeadAndNeckOARs python=3.8.19
conda activate CT_HeadAndNeckOARs
pip install -r requirements.txt  
```
  
## Running pre-trained models  

### Single CT 

```
python run_inference_hn_oars.py <input_path> <session_path> <output_path>
```

### CT cohort   
```
python batch_run_inference_hn_oars.py <input_path> <session_path> <output_path>
```
  
## Citing this work
You may publish material involving results produced using this software provided that you reference the following

1. Iyer, A., Thor, M., Onochie, I., Hesse, J., Zakeri, K., LoCastro, E., ... and Apte, A. P. (2022). Prospectively-validated deep learning model for segmenting swallowing and chewing structures in CT. *Physics in Medicine & Biology*, 67(2), 024001.
2. Jiang, J., Sharif, E., Um, H., Berry, S., and Veeraraghavan, H. (2019). Local block-wise self attention for normal organ segmentation. *arXiv preprint* arXiv:1909.05054.

  
## License
By downloading the software you are agreeing to the following terms and conditions as well as to the Terms of Use of CERR software.

    THE SOFTWARE IS PROVIDED "AS IS" AND CERR DEVELOPMENT TEAM AND ITS COLLABORATORS DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
        
    This software is for research purposes only and has not been approved for clinical use.
    
    Software has not been reviewed or approved by the Food and Drug Administration, and is for non-clinical, IRB-approved Research Use Only. In no event shall data or images generated through the use of the Software be used in the provision of patient care.
    
    You may publish papers and books using results produced using software provided that you reference the appropriate citations (https://doi.org/10.1016/j.phro.2020.05.009, https://doi.org/10.1118/1.1568978, https://doi.org/10.1002/mp.13046, https://doi.org/10.1101/773929)
    
    YOU MAY NOT DISTRIBUTE COPIES of this software, or copies of software derived from this software, to others outside your organization without specific prior written permission from the CERR development team except where noted for specific software products.

    All Technology and technical data delivered under this Agreement are subject to US export control laws and may be subject to export or import regulations in other countries. You agree to comply strictly with all such laws and regulations and acknowledge that you have the responsibility to obtain such licenses to export, re-export, or import as may be required after delivery to you.




