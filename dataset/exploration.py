import csv
import sys
csv.field_size_limit(sys.maxsize)
def count_rows_with_label(csv_file_path, target_label):
    count = 0
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header if it exists
        for row in csv_reader:
            # Assuming the label is in a specific column, adjust the index accordingly
            if target_label in row:
                count += 1

    return count

# Example usage:
csv_file_path = './dataset/train.csv'
target_label = 'spam'
result = count_rows_with_label(csv_file_path, target_label)
print(f'The number of rows containing the label "{target_label}" is: {result}')
