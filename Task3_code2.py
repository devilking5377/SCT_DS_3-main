import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("bank-additional-full.csv", sep=';')

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'y':
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Encode target column
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Select Features and Target
X = df.drop('y', axis=1)
y = df['y']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # You can increase depth for more details
clf.fit(X_train, y_train)

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("Decision Tree for Bank Marketing Prediction", fontsize=16)
plt.show()
