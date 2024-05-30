import csv
import random

# Function to convert a number to its alphabet representation
def number_to_alphabet(number):
    mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E', '6': 'F', '7': 'G', '8': 'H', '9': 'I', '0': 'Z'}
    return ''.join(mapping[digit] for digit in str(number))

# Function to convert a number squared to its alphabet representation
def squared_number_to_alphabet(number):
    squared = number ** 2
    return number_to_alphabet(squared)

# Number of data points
num_data_points = 1000000

# Open the CSV file for writing
with open('dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Original Number', 'Alphabet (Original)', 'Alphabet (Squared)'])
    
    # Generate and write the data
    for _ in range(num_data_points):
        number = random.randint(0,999999)
        alphabet_original = number_to_alphabet(number)
        alphabet_squared = squared_number_to_alphabet(number)
        
        writer.writerow([number, alphabet_original, alphabet_squared])

print("dataset.csv has been created with 1,000,000 rows of data.")
