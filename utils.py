import pandas as pd


# Function that prints the number and percentage of animals in a given column
def print_animal_count(dtf, column_name):
    animal_counts = dtf.groupby(column_name)['animal_id'].count().reset_index(name='count')

    # Calculate the total count of 'animal_id' and the percentage
    total_animals = animal_counts['count'].sum()
    animal_counts['percentage'] = (animal_counts['count'] / total_animals) * 100

    # Sort the result in descending order based on the count of occurrences
    animal_counts_sorted = animal_counts.sort_values(by='count', ascending=False)
    print(animal_counts_sorted)


# Function to convert age strings to years
def convert_to_years(age_str):
    if pd.isnull(age_str):
        return None
    elif 'day' in age_str:
        return int(age_str.split()[0]) / 365
    elif 'week' in age_str:
        return int(age_str.split()[0]) / 52
    elif 'month' in age_str:
        return int(age_str.split()[0]) / 12
    elif 'year' in age_str:
        return int(age_str.split()[0])
    else:
        return None


# Function that returns top k values of an animal in a given column
def get_top_k(dtf, column_name, animal_type, k):
    animal_df = dtf[dtf['animal_type'] == animal_type]
    top_k_values = animal_df[column_name].value_counts().head(k)
    percentages = (top_k_values / len(animal_df)) * 100
    
    # Combine top k values and their percentages into a DataFrame
    top_k_with_percentages = pd.concat([top_k_values, percentages], axis=1)
    top_k_with_percentages.columns = ['count', 'percentage']
    
    return top_k_with_percentages


# Function that only keeps the values of the top k colors of a given animal in a given dataset
def set_top_k_colors(dtf, animal_name, k):
    top_k_colors = get_top_k(dtf, 'color', animal_name, k).index

    # Identify rows with colors not in the top k colors
    not_top_k_colors = dtf[(dtf['animal_type'] == animal_name) & (~dtf['color'].isin(top_k_colors))]

    # Replace the color values for those rows with 'Other color' (a default value)
    dtf.loc[not_top_k_colors.index, 'color'] = 'Other color'
    
    return dtf


# A function that sort colors if both exist
def sort_colors(colors):
    if colors[0] is not None and colors[1] is not None:
        return sorted(colors)
    elif colors[0] is not None:
        return [colors[0], None]


# Function that only keeps the values of the top k breeds of a given animal in a given dataset
def set_top_k_breeds(dtf, animal_name, k):
    top_k_breeds = get_top_k(dtf, 'breed', animal_name, k).index

    # Identify rows with breeds not in the top k breeds
    not_top_k_breeds = dtf[(dtf['animal_type'] == animal_name) & (~dtf['breed'].isin(top_k_breeds))]

    # Replace the breed values for those rows with 'Other breed' (a default value)
    dtf.loc[not_top_k_breeds.index, 'breed'] = 'Other breed'
    
    return dtf
