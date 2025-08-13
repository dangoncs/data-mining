import numpy as np
import pandas as pd


dfX_testToronto = pd.read_csv('X_testToronto.csv')


random_predictions = np.random.choice([0, 1], size=dfX_testToronto.shape[0])


dfX_testToronto['destaque'] = random_predictions

unique, counts = np.unique(random_predictions, return_counts=True)
print("Distribution:", dict(zip(unique, counts)))

dfX_testToronto.to_csv("random.csv", columns=['business_id','destaque'],index=False)
