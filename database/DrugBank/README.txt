Generate the training/validing/testing suits from the DrugBank dataset

These working suits can be generated using the following python scripts 

1. split_sdf.py
   == Split the sdf from one file to separated sdfs 

2. valid_sdf.py
   == Choose the sdfs "only" with the aimming elements
   == 
   ==    a)     Row-based selection of atoms in periodic table can be used 
   ==    b) Element-based selection of atoms in periodic table can be used 
   ==
   == Above selection ways can be implemented by setting "valid_atoms = [xxx]"
   ==

3. group_sdf.py
   == Generate the well-distributed molecular groups basing on the counts of aimming atoms
   == Generate the training/validing/testing suits basing on the ratio like "neach  = [5,1,1]"

4. generate_g09_inp.py 
   == Example for generating inputs file in Gaussian09 format
   == 

