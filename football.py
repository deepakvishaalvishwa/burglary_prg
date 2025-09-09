import numpy as np
import pandas as pd

# Dataset including all attributes
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Football': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                      'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

def entropy(col):
    probs = np.unique(col, return_counts=True)[1] / len(col)
    return -np.sum(probs * np.log2(probs))

def gini(col):
    probs = np.unique(col, return_counts=True)[1] / len(col)
    return 1 - np.sum(probs**2)

def info_gain(df, split_attr, target='Play Football'):
    total_entropy = entropy(df[target])
    vals, counts = np.unique(df[split_attr], return_counts=True)
    weighted_entropy = sum((counts[i]/np.sum(counts)) * entropy(df[df[split_attr] == vals[i]][target]) for i in range(len(vals)))
    return total_entropy, weighted_entropy, total_entropy - weighted_entropy

def gini_gain(df, split_attr, target='Play Football'):
    total_gini = gini(df[target])
    vals, counts = np.unique(df[split_attr], return_counts=True)
    weighted_gini = sum((counts[i]/np.sum(counts)) * gini(df[df[split_attr] == vals[i]][target]) for i in range(len(vals)))
    return total_gini, weighted_gini, total_gini - weighted_gini

# Print entropy and gini for each attribute
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
print(f"Total Entropy of target ('Play Football'): {entropy(df['Play Football']):.3f}")
print(f"Total Gini Index of target ('Play Football'): {gini(df['Play Football']):.3f}\n")

for attr in attributes:
    te, we, ig = info_gain(df, attr)
    tg, wg, gg = gini_gain(df, attr)
    print(f"Attribute: {attr}")
    print(f"  Entropy before split: {te:.3f}")
    print(f"  Entropy after split: {we:.3f}")
    print(f"  Information Gain: {ig:.3f}")
    print(f"  Gini before split: {tg:.3f}")
    print(f"  Gini after split: {wg:.3f}")
    print(f"  Gini Gain: {gg:.3f}\n")
