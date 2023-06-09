# Gender and Societal Bias Measurement and Mitigation for Humanitarian Text Classification

This repository contains the master thesis of Selim Fekih, conducted in EPFL under the supervision of Robert West and Navid Rekabsaz.
## Repository structure
```
├── img                              # Folder containing figures used in the Methodology section in the report. 
|   |── architectures                # Folder containing the model architecture (in .png format)
|   |── metrics                      # Folder containing the different figures of metrics computations (in .png format)
├── results                          
|   |── visualizations               # Folder containing the bias tagwise vizualizations (in .png format)
├── src                              # General folder that containns scripts
|   |── debiasing_pipelines          # Folder that contains the models debiasing scripts
|   |── humbias_set_creation         # Folder that contains the HumSetBias datset creation scripts
|   |── models                       # Folder that contains the models training scripts 
|   |__ pipeline.py                  # Main pipeline script
|   |__ utils.py                     # Util functions for the scripts in the src folder
├── final_results_analysis.py        # Util functions used to postprocess raw outputs and generating the final bias results
├── get_final_results.ipynb          # Notebook for final experiments' results generation 
├── models_training.ipynb            # Notebook for models training and raw outputs generations
├── requirements.txt                 # Set of requirements to be installed for running the project.
├── report.pdf                       # Report of the project
├── LICENSE                          # Project license: MIT
└── README.md
```
## Procedure to reproduce the project's results
 1) Run the `models_training.ipynb` notebook to do the training and generating the raw outputs.
 2) Run the `get_final_results.ipynb` notebook to get the final numerical results and visualizations.
