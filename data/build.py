import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('alpha_combined_cleaned.csv')

# Clean and standardize phenotype labels if needed
df['phenotype'] = df['phenotype'].str.lower().str.strip()

# Set random seed for reproducibility
np.random.seed(42)

# Total desired samples
n_samples = 2000

# Define target class distribution
class_distribution = {
    'normal': int(0.40 * n_samples),           
    'silent carrier': int(0.20 * n_samples),
    'alpha carrier': int(0.25 * n_samples),
    'alpha trait': int(0.15 * n_samples)
}

# Define Gaussian noise levels for each feature
noise_levels = {
    'mcv': 1.0,
    'mch': 0.5,
    'hba2': 0.1,
    'hb': 0.5,
    'rbc': 0.2
}

# Define biologically valid ranges (clipping thresholds)
clip_ranges = {
    'mcv': (55, 95),
    'mch': (18, 35),
    'hba2': (1.5, 5.5),
    'hb': (5, 17),
    'rbc': (2.5, 6.8)
}

# Gaussian bootstrapping function
def gaussian_bootstrap(df_class, n_target):
    sampled = df_class.sample(n=n_target, replace=True).reset_index(drop=True)
    for feature in noise_levels:
        sampled[feature] += np.random.normal(0, noise_levels[feature], n_target)
        sampled[feature] = sampled[feature].clip(*clip_ranges[feature])
    return sampled

# Bootstrap each class
bootstrapped_dfs = []
for phenotype, target_count in class_distribution.items():
    df_class = df[df['phenotype'] == phenotype]
    bootstrapped_df = gaussian_bootstrap(df_class, target_count)
    bootstrapped_dfs.append(bootstrapped_df)

# Combine all bootstrapped samples and shuffle
final_df = pd.concat(bootstrapped_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Save the final bootstrapped dataset
final_df.to_csv('alpha_thalassemia_gaussian_bootstrapped_dataset.csv', index=False)

print("Bootstrapped dataset generated with 2000 samples!")
