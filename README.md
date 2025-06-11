# PhD Thesis Project -- QSAR-Guided Design and Synthesis of Novel Inhibitors Targeting KRAS Protein-Protein Interactions

Training and evaluation of ten regression algorithms on in-house library of KRAS protein-protein interaction inhibitors. The best model was selected to predict activities of the molecules from several molecular libraries regarding inhibition kras nucleotide exchange.
# Python QSAR Pipeline for Screening of novel KRAS Protein-Protein Interaction Inhibitors (PPIIs)

## General Information
This project was inspired by the pubication RSC Med. Chem., 2024, 15,
1392 (DOI: 10.1039/D4MD00063C) by Duo _et al_., which can be found here:
https://pubs.rsc.org/en/content/articlelanding/2024/md/d4md00063c.
I used the code associated with the publication as a starting point for this project. The code can be found here:
https://github.com/cristinaduo/ML-for-SOS1
The code has been adapted significantly for this project. In especially, nested cross-validation was implemented as well as a feature selection pipeline.

## Data preparation and Analysis
Knime was used to prepare the structures and target variables. [Data preparation workflow](knime_.knwf)


Two anaconda environments were used:

## Description
### Data Preparation
This project was created during my PhD studies in the Scherkenbeck group at the BUW.
A library of ~600 KRAS PPIIs with associated activities was accumulated over the past decade in the Scherkenbeck group.
My colleage Sascha Koller and I decided to extract quantitative structure activity relationships (QSARs) from the library.
We washed the molecular structures of the library in KNIME and manually curated the assay data. We decided to use IC$_{50}$ values of a nucleotide exchange assay as target values for the employed regression algorithms. The assay protocol can be found in this publication by Benary _et al_.: RSC Adv.,2025, 15, 883 (DOI: 10.1039/D4RA08503E).
The in-house library was extended using SAR information of known, non-covalent KRAS PPIIs from the ChEMBL34 database. The structures were washed and the assay data cleaned analogous to our in-house library. TSNE plots of our in-house library and the in-house+ChEMBL-extension libraries are shown below:

_Insert TSNE plots here_

The plots highlight that the ChEMBL extension added a significant number of highly active KRAS PPIIs to the combined library. By combining the training dataset was extended from 603 to 917 structures. The training data is not included in this project.

### Feature selection
Duo _et al._ based their QSAR solely on ECFP of radius 3 and length 512. In this project a significant improvement in the R^2 values of the employed regression algorithms was observed when 20 molecular descriptors were used as features additionally. All 2D descriptors available in the Mordred software package were calculated for the training data. The Mordred software can be found here: https://github.com/mordred-descriptor/mordred?tab=readme-ov-file. The descriptors were ranked according to their mutual information with the target values. The 20 best were selected.

_Insert list of 20 best descriptors_

### Algorithm selection
The performance of ten regression algorithms was tested on the training data using nested cross-validation.

* RandomForestRegressor
* AdaBoostRegressor
* GradientBoostingRegressor
* ElasticNet
* Lasso
* DecisionTreeRegressor
* SupportVectorRegressor
* Nearest Neighbors
* Ridge
* ExtraTreesRegressor

A hyperparameter grid search with five inner folds was implemented for each of five outer folds. The mean and standard deviation of the five interations are summarised for each model in the following table. The algorithms are ranked ascendingly according to the test R^2 values.

_Insert table here_

Learning curves were plotted for the three best performing algorithms.

_Insert plots here_

Of all ten tested algotirhms the RandomForestRegressor (_insert exact settings here_) performed best. The hyperparameter search and fitting was repeated on 80% of the training data. The obtained validation model was used to predict activity values for the remaining test set of 20%. Manual inspection of the prediction results indicated sufficient accuracy for application of the model. The hyperparameter search and fitting was repeated on the whole training data. All models and prediction can be found in the respective folders.

### Activity prediction of molecular libraries
The developed QSAR model was used to predict activities for the following libraries. Each library is listed along with its download source.

* ChEMBL34: https://chembl.gitbook.io/chembl-interface-documentation/downloads
* Enamine Screening: https://enamine.net/compound-collections/screening-collection
* Enamine PPI: https://enamine.net/compound-libraries/targeted-libraries/ppi-library
* BMS300k: https://www.chemdiv.com/catalog/diversity-libraries/bemis-murcko-clustering-library/
* iPPI: https://ippidb.pasteur.fr/
* ReinventPPI: https://github.com/ohuelab/iPPI-REINVENT
* npatlas: https://www.npatlas.org/

Additionally a library of all achiral cyclic tetrapeptides from 20 proteinogenic amino acids was generated in KNIME. A library of click cyclic tetrapeptides was generated analogously.
All structures in the libraries were washed in KNIME analogously to the training structures. The training set was subtracted from each of the prediction libraries.
