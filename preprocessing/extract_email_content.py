import os
import csv

def extract_after_first_line(directory_path):
    result_list = []
    html_count = 0

    # Iterate over files in the specified directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if the file is a regular file and not a directory
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

                subject_pos = content.lower().find('subject:')

                if subject_pos != -1:
                    # Find the position of the next line break after "subject:"
                    line_break_pos = content.find('\n', subject_pos)

                    # Extract everything after "subject:" until the next line break
                    if line_break_pos != -1:
                        extracted_subject = content[subject_pos + len('subject:'):line_break_pos].strip()
                       
                # Find the position of the first line break
                empty_line_pos = content.find('\n\n')

                if empty_line_pos != -1:
                    extracted_content = content[empty_line_pos + 2:]  # Skip the two line breaks
                    result_list.append({'Subject': extracted_subject, 'Content': extracted_content})

                if '<html>' in content.lower():
                    html_count += 1

    return result_list, html_count

directory_path = '../dataset/spam_2'
result, html_count = extract_after_first_line(directory_path)

csv_file_path = 'cleaned/spam.csv'
fields = ['Subject', 'Content']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=fields)
    
    # Write the header
    csv_writer.writeheader()

    # Write the data
    for row in result:
        csv_writer.writerow(row)

print(html_count)