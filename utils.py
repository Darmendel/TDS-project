import pandas as pd


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


# Function that prints the number and percentage of animals in a given column
def print_animal_count(dtf, column_name):
    animal_counts = dtf.groupby(column_name)['animal_id'].count().reset_index(name='count')

    # Calculate the total count of 'animal_id' and the percentage
    total_animals = animal_counts['count'].sum()
    animal_counts['percentage'] = (animal_counts['count'] / total_animals) * 100

    # Sort the result in descending order based on the count of occurrences
    animal_counts_sorted = animal_counts.sort_values(by='count', ascending=False)

    print(animal_counts_sorted)


# A function that sort colors if both exist
def sort_colors(colors):
    if colors[0] is not None and colors[1] is not None:
        return sorted(colors)
    elif colors[0] is not None:
        return [colors[0], None]
