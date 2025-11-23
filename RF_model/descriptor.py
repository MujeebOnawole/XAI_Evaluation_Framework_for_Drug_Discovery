import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Read the input CSV file
input_file = 'S_aureus_cleaned_data.csv'  # Replace with your input file name
df = pd.read_csv(input_file)

# Define a function to calculate descriptors
def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES strings gracefully
    vals = Descriptors.CalcMolDescriptors(mol)
    return {key: val for key, val in vals.items() if key.startswith('fr_')}

# Calculate descriptors for each row
descriptor_dicts = df['PROCESSED_SMILES'].apply(calc_descriptors)

# Create a DataFrame from the descriptor dictionaries
descriptor_df = pd.DataFrame(list(descriptor_dicts))

# Combine the descriptor DataFrame with the original columns
final_df = pd.concat([df[['COMPOUND_ID', 'PROCESSED_SMILES', 'TARGET', 'group']], descriptor_df], axis=1)

# Display the first few rows of the final DataFrame
print(final_df.head())

# Optionally, save the final DataFrame to a new CSV file
output_file = 'SA_FG_fragments.csv'  # Replace with your desired output file name
final_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")