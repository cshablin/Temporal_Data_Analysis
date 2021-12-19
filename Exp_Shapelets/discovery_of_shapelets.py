import pandas as pd
from gendis.genetic import GeneticExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read in the datafiles
train_df = pd.read_csv('<DATA_FILE>')
test_df = pd.read_csv('<DATA_FILE>')
# Split into feature matrices and label vectors
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# Univariate time series example
# Creating a GeneticExtractor object
genetic_extractor = GeneticExtractor(population_size=50, iterations=25, verbose=True,
                                     mutation_prob=0.3, crossover_prob=0.3,
                                     wait=10, max_len=len(X_train) // 2)

# Fit the GeneticExtractor and construct distance matrix
shapelets = genetic_extractor.fit(X_train, y_train)
distances_train = genetic_extractor.transform(X_train)
distances_test = genetic_extractor.transform(X_test)

# Fit ML classifier on constructed distance matrix
lr = LogisticRegression()
lr.fit(distances_train, y_train)

print('Accuracy = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))