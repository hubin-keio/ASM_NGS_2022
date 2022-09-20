# ASM_NGS_2022

Source code used for the ASM NGS Conference (2002) Talk

Control/Tracking Number: 2022-A-74-NGS

## Using Deep Mutational Data and Machine Learning to Guide Outbreak and Pandemic Response

Bin Hu, Michal Babinski, and Patrick Chain
    
Los Alamos National Laboratory

### Abstract
    
A significant fraction of pathogens known to infect humans originate in non-human (zoonotic) hosts and new and emerging pathogens continue to spill over into the human population more frequently at an alarming rate (e.g., SARS, MERS, Cholera, etc.). The recent outbreaks of Ebola virus in West Africa and the ongoing SARS-CoV-2 pandemic demonstrate the need for rapid and reliable assessments of viral phenotype information to help inform scientists and policy makers how best to control the spread of disease. Further understanding of the virus pathogenic evolutionary space and potential trajectory could guide appropriate control measures to limit the spread of a new virus throughout the local and global human population.

Once a viral disease begins to circulate, reliable diagnostics, protective vaccines and therapeutic antibodies are essential tools for preventing, monitoring, and managing disease spread. However, the efficacy of these tools can be diminished by mutations in viral genomes, and the delay between the emergence of new viral strains and redesign of vaccines and diagnostics allows for continued viral transmission. Given the combinatorial explosion of potential mutations that could enable a virus to “escape” diagnostics, vaccines and antibodies, and the high cost of biomedical research, it is essential to focus countermeasure development efforts only on viral strains that pose the highest risk to society. Towards this end, the questions we ask are: Is it possible to predict the most likely evolutionary trajectory of circulating genomes and anticipate novel variants before they emerge? Is it possible to assess the risk of future variants by computationally predicting key virulence determinants and exploring the evolutionary space for pathogenicity?

To address these questions using the machine learning approach, we have developed several neural networks using deep mutational scanning (DMS) data. This simple model was able to predict fairly accurately the RBD expression and binding to ACE2 (R2 = 0.76). It only takes less than a second for the model to predict the effect of an arbitrary mutation, regardless of the combinatorial complexity, on a consumer PC. Recently, principal component analysis of amino acid biochemical properties and graph neural network (GNN) to learn protein properties have been combined to predict antibody binding and enzyme activities in five proteins (Gelman et al. 2021). Using a similar approach, we have developed a GNN model to study RBD and are currently evaluating the model. Combining DMS and deep learning, we can predict the mutational effects of SARS-CoV-2 RBD, both in its expression and binding to the ACE2 receptor. We are evaluating different ML models and will deploy either the best model or an ensemble of models to our existing SARS-CoV-2 sequence monitoring workflow. The upgraded workflow will be able to rank the latest mutations based on potential threat level. 

