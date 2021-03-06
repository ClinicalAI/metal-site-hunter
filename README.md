# MetalSiteHunter

We offer an ensemble 3D deep convolution neural network to predict the metal binding sites. We consider protein structures as three-dimensional voxels parameterized by atoms' biophysical properties. We exploit the strength of the CNN architecture to detect spatially proximate features. The overall pipeline can be found in the following figure:

![metal-site-prediction-data-pipeline](https://github.com/ClinicalAI/metal-site-hunter/blob/main/model_pipe_line.png)
## Data
We used the MetalPDB database to collect the 3D structures of the protein metal binding sites. We built the 3D voxels from the protein binding sites using [HTMD](https://software.acellera.com/docs/latest/htmd/index.html) python package. For each pocket in our dataset we load the PDB structures and apply the prepareProteinForAtomtyping to remove non-protein atoms and add polar hydrogens. We then developed a Python code to calculate the center of the pocket by averaging the coordinations of all alpha-carbon available in the structure. We selected five different type of voxel from each pocket namely, ‘positive_ionizable’, ‘hbond_acceptor’, ‘hbond_donor’, ‘occupancies’ and ‘negative_ionizable’ to train our model. We finally built the 20x20x20 voxels for each binding site to train a 3D deep learning model. The data of these voxels can be found in the [Data](https://drive.google.com/drive/folders/1CG7o3m7hTQV44WRC6jX3QkH0FpNt5Blb?usp=sharing) folder as npz files. 
## Codes
To train the base models you can use the following code:

[base_models.py](https://github.com/ClinicalAI/metal-site-hunter/blob/main/base_models.py)

After training the base models, we save the weights. Then we load both model1 and model2 and ensemble them, followed by fully connected layers. The code for the ensemble models can be found here:

[ensemble.py](https://github.com/ClinicalAI/metal-site-hunter/blob/main/ensemble.py)

We finally evaluate our models on the unseen test data:

[evaluate.py](https://github.com/ClinicalAI/metal-site-hunter/blob/main/evaluate.py)



## Web-server:

We built a [web-server](https://mohamad-lab.ai/metalsitehunter/) based on our models. You can try our model there!



