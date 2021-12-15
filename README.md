# metal-site-prediction

We offer an ensemble 3D deep convolution neural network to predict the metal binding sites. We consider protein structures to be three-dimensional images with voxels parameterized by atom biophysical properties. We exploit the strength of CNN architecture to detect spatially proximate features. These detectors of local biochemical interactions are then hierarchically composed into more intricate features capable of describing the complex and nonlinear phenomenon of molecular interaction. We finally used the features in an ensemble model to predict the metal-binding site. you can use google colab to repoduce the results or using this [web server](https://mohamad-lab.ai/metal-prediction/ "Metal Site Prediction") for testing. 

![metal-site-prediction-data-pipeline](https://github.com/ClinicalAI/metal-site-hunter/blob/main/metal_prediction_pipeline.png)
## Data
We used MetalPDB and RCSB database to collect the 3D structure of the metalloproteins and non-metal PDBS. As MetalPDB compiled all the binding sites of a single protein, some of the binding sites are identical structures that occurred in different domains of the same protein. We used CD-HIT (parameters: ) to remove the redundant structures with more than 90% similarity. We found 20,972 of these structures are unique among total 243,600 (Table1).  We selected the top three metals (Mg, Zn, and Fe) which has enough samples (more than 3,000) to train the models to extract features.
We extracted 3D features using [HTMD](https://software.acellera.com/docs/latest/htmd/index.html) python package.  We used getVoxelDescriptors of HTMD package to build the 3D feature descriptors for each PDB file. For each pocket in our dataset we load the PDB structures and apply the prepareProteinForAtomtyping to remove non-protein atoms and add polar hydrogens. We then developed a Python code to calculate the center of the pocket by averaging the coordinations of all alpha-carbon available in the structure. We generated seven different type of voxel for each pocket namely, ‘hydrophobic’, ‘aromatic’, ‘hbond_acceptor’, ‘hbond_donor’, ‘positive_ionizable’, ‘negative_ionizable’ and ‘occupancies’. We removed those structures which HTMD couldn’t build 3D voxel for them from our dataset. Finally, we built 3D voxel structures for the 10,383 structures available in our datasets.

## Codes
This project used Google Colab for processing, you can use the note-books in _Codes_ directory and run them on the Google Colab.
## Pre-Trained Models
Also there are set of pre-trained models that we used in our web server in the _Trained_models_ directory.
## Training Models
The note-books under the _Codes_ directory can be tested and run. By default, this note-books train on our datasets in the _Data_ directory, while the ensemble model trains on our pre-trained models in the _Trained Models_ directory. However, if you want to use different models, you must first run _metal_site_prediction_base_models.ipynb_ and save the results to your Google Drive before using these pre-trained models in _metal_site_prediction_ensemble_model.ipynb_ for training on your selected model set.
