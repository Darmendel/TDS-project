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


# Function that returns top k values of an animal in a given column
def get_top_k(dtf, column_name, animal_type, k):
    animal_df = dtf[dtf['animal_type'] == animal_type]
    top_k_values = animal_df[column_name].value_counts().head(k)
    percentages = (top_k_values / len(animal_df)) * 100
    
    # Combine top k values and their percentages into a DataFrame
    top_k_with_percentages = pd.concat([top_k_values, percentages], axis=1)
    top_k_with_percentages.columns = ['count', 'percentage']
    
    return top_k_with_percentages


# # Function that returns top k values of an animal in a given column
# def get_top_k(dtf, column_name, type, k):
#     vals = dtf[dtf['animal_type'] == type][column_name]
#     return vals.value_counts().head(k)


# # Function that returns top k breeds of an animal
# def get_top_k_breeds(dtf, type, k):
#     breeds = dtf[dtf['animal_type'] == type]['breed']
#     return breeds.value_counts().head(k)


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


# A function that sort colors if both exist
def sort_colors(colors):
    if colors[0] is not None and colors[1] is not None:
        return sorted(colors)
    elif colors[0] is not None:
        return [colors[0], None]
