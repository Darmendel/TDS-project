# Function that prints all the id's of animals who have missing values in a given column
def print_animal_ids(column_name, animal_ids, dataset_name = ''):
    if dataset_name != '':
        print(f"Animal IDs with missing values in '{column_name}' column that exist in {dataset_name} dataset:"
      f"\n{animal_ids}")
    else:
        print(f"Animal IDs with missing values in '{column_name}' column:\n{animal_ids}\n")


# Function that returns all 'animal_id' of rows with None values in a given column
def get_missing_animal_ids(dtf, column_name):
    return dtf.loc[dtf[column_name].isnull(), 'animal_id']


# Function that return rows from a dataset that have non-none values in a given column in dtf
def get_existing_animal_ids_in_dataset(dataset, column_name, missing_animal_ids, id):
    df_not_null = dataset[dataset[column_name].notnull()]
    exist_in_df = missing_animal_ids.isin(df_not_null[id])
    existing_animal_ids = missing_animal_ids[exist_in_df]
    return existing_animal_ids


# Function that filters out the id's of animals who have missing values in a given column in the dtf dataset
# and also have missing values in a given column in a given dataset (df_in_out or df_in)
def filter_null_values(dataset, column_name, existing_animal_ids, values_list, id):
    ids_to_remove = []

    for animal_id in existing_animal_ids:
        index_found = dataset[dataset[id] == animal_id].index[0]
        if str(dataset.loc[index_found, column_name]) not in values_list:
            ids_to_remove.append(animal_id)
            continue
    
    # Iterate over each element in ids_to_remove and remove it from existing_animal_ids
    for animal_id in ids_to_remove:
        existing_animal_ids = existing_animal_ids[existing_animal_ids != animal_id]
    
    return existing_animal_ids


# Function that retrieves the missing values in a given column from a given dataset
# and updates the column in the dtf dataset with the new values
def update_column(dtf, dataset, column_name, existing_animal_ids, id):
    animal_ids = [id for id in existing_animal_ids]

    # Update the column for all animal_ids
    for animal_id in animal_ids:
        index_found = dataset[dataset[id] == animal_id].index[0]
        new_value = dataset.loc[index_found, column_name]
        dtf.loc[dtf['animal_id'] == animal_id, column_name] = new_value
    
    return dtf
