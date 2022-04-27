- database : 
  - DrugBank : DrugBank all-in-one sdf file, and the scripts for generating the training and the testing suits
  - polyfitted : Fitted ploynormal equations for selected 89 reference DFT functionals
  - rawdata : The assembled Gaussuian09-D.01 timing data, and the separated sdf files with added H atom
      - for the data that is organized as follows : 
      - 126  0  81.7  0  347828991  13  6.28461538462  2097152000  6-31G  [126, 126]  E(RB3LYP)  [57, 57]  ['-0.032913667446', '-0.203441313753', '-0.000529993487', '-0.000486078130', '-0.000010937382', '0.000126085332', '-0.000000096350', '-0.000000178336', '-0.000000012795', '-0.000000000448', '-0.000000000047', '0.000000000019'] [uncontracted : 140 168 0 0 0 0] [  contracted : 42 84 0 0 0 0]
        - 126 -- total number of atomic orbitals
        - 0 -- Not used yet
        - 81.7 -- Total time for this calculation (seconds)
        - 0 -- Not used yet 
        - 347828991 -- ID of SDF in Drugbank
        - 13 -- Number of SCF iterations
        - 6.28461538462 -- Time for each iteration (seconds)
        - 
  - trained-models : Trained models for few DFT functional/basis set combinations



