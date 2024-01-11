import seaborn as sns
import matplotlib.pyplot as plt

# Your key-value pairs
data = {'Naive Bayes': 0.95, 'BERT 1 epoch': 0.96, 'BERT 2 epochs': 0.97, 'BERT 3 epochs': 0.98}

# Convert the data to lists
categories = list(data.keys())
values = list(data.values())

# Create a bar chart using Seaborn
sns.barplot(x=categories, y=values)

plt.ylim(0.9, 1)

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Naive Bayes and BERT with ascending training epochs')

# Show the plot
plt.show()
