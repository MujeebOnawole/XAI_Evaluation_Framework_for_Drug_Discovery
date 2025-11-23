import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.ExactMolWt(mol) if mol else None

def classify_molecule(smiles, mol_wt, max_mol_wt):
    molecule = Chem.MolFromSmiles(smiles)
    if not molecule:
        return 'cannot be processed by RDKit'
    if '.' in smiles:
        return 'mixture or salt'
    if all(atom.GetSymbol() != 'C' for atom in molecule.GetAtoms()):
        return 'inorganic compound'
    if mol_wt > max_mol_wt:
        return 'molecular weight exceeds limit'
    return 'valid'

def filter_data(data, smiles_column, max_mol_wt=700):
    data['MOL_WT'] = data[smiles_column].apply(calculate_molecular_weight)
    data['REASON'] = data.apply(lambda x: classify_molecule(x[smiles_column], x['MOL_WT'], max_mol_wt), axis=1)
    return data[data['REASON'] == 'valid'], data[data['REASON'] != 'valid']

def dataset_cleaning(file_name, output_prefix):
    # Load the dataset
    data = pd.read_csv(file_name)
    origin_data_num = len(data)

    # Filter data using the classification function
    valid_data, invalid_data = filter_data(data, 'PROCESSED_SMILES')
    print(f"Filtered data contains {len(valid_data)} valid records.")
    print(f"Filtered out {len(invalid_data)} records for various reasons.")

    # Separate compounds with max_similarity = 1
    reference_compounds = valid_data[valid_data['max_similarity'] == 1]
    other_compounds = valid_data[valid_data['max_similarity'] != 1]

    # Calculate the number of samples needed for each set
    total_samples = len(valid_data)
    train_samples = int(0.8 * total_samples)
    valid_samples = int(0.1 * total_samples)
    test_samples = total_samples - train_samples - valid_samples

    # Ensure all reference compounds are in the training set
    train_data = reference_compounds.copy()
    remaining_train_samples = train_samples - len(train_data)

    # Stratified split of the remaining compounds
    if remaining_train_samples > 0:
        temp_train, temp_rest = train_test_split(other_compounds, 
                                                 train_size=remaining_train_samples, 
                                                 stratify=other_compounds['TARGET'], 
                                                 random_state=42)
        train_data = pd.concat([train_data, temp_train])
        valid_data, test_data = train_test_split(temp_rest, 
                                                 test_size=test_samples/(valid_samples + test_samples), 
                                                 stratify=temp_rest['TARGET'], 
                                                 random_state=42)
    else:
        # If we have more reference compounds than needed for training,
        # we need to adjust the split to maintain the original distribution
        train_data, temp_rest = train_test_split(valid_data, 
                                                 train_size=train_samples, 
                                                 stratify=valid_data['TARGET'], 
                                                 random_state=42)
        valid_data, test_data = train_test_split(temp_rest, 
                                                 test_size=test_samples/(valid_samples + test_samples), 
                                                 stratify=temp_rest['TARGET'], 
                                                 random_state=42)
        
        # Ensure all reference compounds are in the training set
        reference_in_valid = valid_data[valid_data['max_similarity'] == 1]
        reference_in_test = test_data[test_data['max_similarity'] == 1]
        
        if len(reference_in_valid) > 0 or len(reference_in_test) > 0:
            train_data = pd.concat([train_data, reference_in_valid, reference_in_test])
            valid_data = valid_data[valid_data['max_similarity'] != 1]
            test_data = test_data[test_data['max_similarity'] != 1]
            
            # Rebalance the sets
            excess = len(train_data) - train_samples
            move_to_valid, move_to_test = train_test_split(
                train_data[train_data['max_similarity'] != 1].sample(n=excess, random_state=42),
                test_size=test_samples/(valid_samples + test_samples),
                stratify=train_data[train_data['max_similarity'] != 1].sample(n=excess, random_state=42)['TARGET'],
                random_state=42
            )
            
            train_data = train_data[~train_data.index.isin(move_to_valid.index) & ~train_data.index.isin(move_to_test.index)]
            valid_data = pd.concat([valid_data, move_to_valid])
            test_data = pd.concat([test_data, move_to_test])

    # Add the group column
    train_data['group'] = 'training'
    valid_data['group'] = 'valid'
    test_data['group'] = 'test'

    # Combine all data back together
    full_data = pd.concat([train_data, valid_data, test_data])

    # Save the combined dataset to CSV file
    full_data.to_csv(f'{output_prefix}.csv', index=False)

    # Save the excluded compounds to a CSV file
    invalid_data.to_csv(f'{output_prefix}_filtered_out.csv', index=False)

    # Print and save the distribution of TARGET in each set
    def print_and_save_distribution(dataset, name, log_file):
        distribution = dataset['TARGET'].value_counts(normalize=True)
        log_file.write(f"{name} set distribution:\n{distribution}\n\n")
        print(f"{name} set distribution:\n{distribution}\n")

    with open(f'{output_prefix}_log.txt', 'w') as log_file:
        log_file.write('Data cleaning is over!\n\n')
        print_and_save_distribution(valid_data, "Original", log_file)
        print_and_save_distribution(train_data, "Training", log_file)
        print_and_save_distribution(valid_data, "Validation", log_file)
        print_and_save_distribution(test_data, "Test", log_file)

    print(f"Original data: {origin_data_num} molecules")
    print(f"After cleaning: {len(full_data)} molecules")
    print(f"Training set: {len(train_data)} molecules")
    print(f"Validation set: {len(valid_data)} molecules")
    print(f"Test set: {len(test_data)} molecules")

# Call the function with your dataset
dataset_cleaning('similarity_result.csv', 'cleaned_data')