import subprocess
import copy
import os
import shutil
import random
from urllib import request
from Bio.PDB import *

non_metals = []
pdbs = open('pdbs.txt').read().splitlines()
pdbs = [sample.replace(' ', '') for sample in pdbs]
random.shuffle(pdbs)

parser = PDBParser(PERMISSIVE=True, QUIET=True)


def metal_bind(pdb):

  metals = ['ZN', 'FE', 'CU', 'MG', 'MN','NA']

  structure_id = pdb.split('/')[-1].split('.')[0]
  structure = parser.get_structure(structure_id, pdb)
  atoms_name = []

  for atom in structure.get_atoms():
    atoms_name.append(atom.name)

  for metal in metals:
    if metal in atoms_name:
      return True
  return False


for index,pdb in enumerate(pdbs):
  try:
      print('protein number    ',index)
      print('number of new non metal    ' ,len(non_metals))
      
      pdb_dir = f'pdbs/{pdb}/{pdb}.pdb'
      subprocess.run(['mkdir','-p', f'pdbs/{pdb}'],check=True)
      request.urlretrieve(f'http://files.rcsb.org/download/{pdb}.pdb', pdb_dir)
      
      if not metal_bind(pdb_dir):
        subprocess.run(['fpocket','-f', pdb_dir],check=True)
        pocket = f'pdbs/{pdb}/{pdb}_out/pockets/pocket1_atm.pdb'
        
        pocket_size = os.path.getsize(pocket)
        
        if 13192 < pocket_size < 36384 :
          out_dir = f'/content/drive/MyDrive/data/non_metal_pockets/{pdb}.pdb'
          subprocess.run(['cp', pocket, out_dir ], check=True)
          non_metals.append(pdb)
          
        else:
          shutil.rmtree(f'pdbs/{pdb}')
      else:
        shutil.rmtree(f'pdbs/{pdb}')
  except:
    pass
  
  if len(non_metals) > 3000:
    break

