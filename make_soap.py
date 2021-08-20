from ase.io import read
from ase import Atoms
from dscribe.descriptors import SOAP
import numpy as np
import os
import fileinput

#create a list of structures and gather the chemical elements
sdf = open("train_mols.sdf","r") #change to test_mols.sdf
sdf_contents = sdf.readlines()

structures = []
structure=[]

for line in sdf_contents:
    if line !="$$$$\n":
        structure.append(line)
    else:
        structure.append(line)
        # convert structure to format required
        structure_sdf_string=''.join(structure)
        temp_sdf_file=open("temp.sdf","w")
        temp_sdf_file.write(structure_sdf_string)
        temp_sdf_file.close()
        structure_read = read("temp.sdf")
        structures.append(structure_read)
        structure=[]
        continue
sdf.close()
os.remove(os.getcwd()+"/temp.sdf")

#find elements in dataset for SOAP
species = set()
for structure in structures:
    species.update(structure.get_chemical_symbols())
print(species)

#configure SOAP descriptor
soap=SOAP(
    species=species,
    periodic=False,
    rcut=6.0,
    nmax=4,
    lmax=4,
    average="outer",
    sparse=False
)

#create SOAP feature vectors for each structure
feature_vectors = soap.create(structures,n_jobs=1)
print(np.shape(feature_vectors))
cols = np.shape(feature_vectors)[1]

#create header for output
header=[]
for col in range(cols):
    header.append('soap{}'.format(col+1))
header_str = ','.join(header)

#save array to csv file
data = np.asarray(feature_vectors)
np.savetxt('soap.csv',data,delimiter=',')

#add in header to csv file
csv_file = open('soap.csv',"r")
csv_contents=csv_file.read()
csv_file.close()
new_contents=header_str+"\n"+csv_contents
csv_file = open('soap.csv',"w")
csv_file.write(new_contents)
csv_file.close()
