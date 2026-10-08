import os
import re
import shlex
import pickle
import numpy as np
import pandas as pd
from ase import Atoms

def parse_xyz_to_dataframe(file_path):
    """Reads an extended XYZ file and parses it into a pandas DataFrame."""
    print(f"📖 Reading configuration target: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find target file at: {os.path.abspath(file_path)}")
        return None

    with open(file_path, 'r') as raw_data:
        file_content = raw_data.read()

    # Split string blocks cleanly right before structural sections
    pattern = r'(?:^|\n)(\d+\n?Lattice=[\s\S]*?)(?=\n\d+\n?Lattice=|$)'
    chunks = re.findall(pattern, file_content)
    print(f"   -> Found {len(chunks)} raw configuration entries.")
    
    rows = []
    for chunk in chunks:
        data = chunk.strip().split('\n')
        natoms = data[0]          
        metadata_line = data[1]   
        pos = data[2:]            
        
        data_dict = {
            'natoms': int(natoms), 
            'positions': pos       
        }
        
        parsed_elements = shlex.split(metadata_line)
        for element in parsed_elements:
            if '=' in element:
                key, value = element.split('=', 1)
                data_dict[key] = value

        rows.append(data_dict)

    return pd.DataFrame(rows)

def build_ase_dataset(df):
    """Processes raw configuration lines into native ASE Atoms and NumPy metrics."""
    if df is None or df.empty:
        return pd.DataFrame()

    pace_rows = []
    for index, row in df.iterrows():
        try:
            # Parse Unit Cell Matrix
            lattice_vals = np.fromstring(row['Lattice'], sep=' ')
            cell_matrix = lattice_vals.reshape((3, 3))
            
            # Parse Periodic Boundary Conditions (PBC)
            pbc_input = row['pbc']
            if isinstance(pbc_input, str):
                pbc_list = [token.strip().upper() in ['T', 'TRUE', '1'] for token in pbc_input.split()]
            else:
                pbc_list = [str(token).strip().upper() in ['T', 'TRUE', '1'] for token in pbc_input]
                
            symbols = []
            positions = []
            magmoms = []
            forces = []
            
            # Unpack individual text matrix lines
            for atom_line in row['positions']:
                parts = atom_line.split()
                if not parts:
                    continue
                symbols.append(parts[0])                                                    
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])        
                magmoms.append(float(parts[4]))                                             
                forces.append([float(parts[5]), float(parts[6]), float(parts[7])])           
                
            # Build and seed native object
            atoms_obj = Atoms(
                symbols=symbols,
                positions=positions,
                cell=cell_matrix,
                pbc=pbc_list
            )
            atoms_obj.set_initial_magnetic_moments(magmoms)

            total_energy = float(row['energy'])
            num_atoms = len(symbols)
            energy_per_atom = total_energy / num_atoms if num_atoms > 0 else 0.0

            pace_entry = {
                'ase_atoms': atoms_obj,                  
                'energy': total_energy,                  
                'energy_per_atom': energy_per_atom,      
                'forces': np.array(forces),              
                'structure_name': row.get('structure_name', f'struct_{index}') 
            }
            
            # Handle stress tensor variations
            if 'stress' in row and isinstance(row['stress'], str):
                stress_vals = np.fromstring(row['stress'], sep=' ')
                if len(stress_vals) == 9:
                    pace_entry['stress'] = stress_vals.reshape((3, 3))
                elif len(stress_vals) == 6:
                    pace_entry['stress'] = stress_vals
                    
            pace_rows.append(pace_entry)
            
        except Exception as e:
            print(f"⚠️ Skipping configuration index {index} due to processing error: {e}")

    return pd.DataFrame(pace_rows)

def main():
    # Target configurations mapping explicitly to your project layouts
    TARGETS = [
        {"input": "train.xyz", "output": "training.pkl"},
        {"input": "val.xyz",   "output": "testing.pkl"}
    ]

    for target in TARGETS:
        print(f"\n⚡ Processing file sequence: {target['input']} -> {target['output']}")
        
        # 1. Unpack raw structures into dataframe matrix rows
        raw_df = parse_xyz_to_dataframe(target['input'])
        
        # 2. Build mathematical representation structures using NumPy 1.x
        processed_df = build_ase_dataset(raw_df)
        
        if not processed_df.empty:
            print(f"💾 Exporting cluster-native database to: {target['output']}")
            processed_df.to_pickle(target['output'])
            print(f"✅ Finished writing {len(processed_df)} records!")
        else:
            print(f"❌ skipping write phase for {target['output']} due to processing collapse.")

    print("\n🏁 Dataset preparation finalized successfully!")

if __name__ == "__main__":
    main()

