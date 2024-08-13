import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def calculate_statistics(df):
    stats = df.describe().T
    # stats['variance'] = df.var()
    # stats['range'] = df.max() - df.min()
    return stats

# Function to apply standard scaling
# Function to apply standard scaling only to numeric columns, excluding columns with inf or too large values
def apply_standard_scaler(datasets):
    scaler = StandardScaler()
    scaled_datasets = {}
    problematic_columns = {}

    for name, df in datasets.items():
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        
        # Identify and exclude columns with inf or very large values
        cols_to_scale = []
        cols_excluded = []
        
        for col in numeric_cols:
            if np.isfinite(df[col]).all() and np.abs(df[col]).max() < np.finfo(np.float64).max:
                cols_to_scale.append(col)
            else:
                cols_excluded.append(col)
        
        if cols_excluded:
            problematic_columns[name] = cols_excluded
        
        # Apply scaler only to safe numeric columns
        scaled_numeric_data = scaler.fit_transform(df[cols_to_scale])
        scaled_df = pd.DataFrame(scaled_numeric_data, columns=cols_to_scale, index=df.index)
        
        # Combine scaled numeric data with non-numeric data
        final_df = pd.concat([scaled_df, df[non_numeric_cols]], axis=1)
        scaled_datasets[name] = final_df

    return scaled_datasets, problematic_columns

def plot_distributions(datasets, selected_indicator):
    combined_df = pd.concat(
        [df[[selected_indicator]].assign(Dataset=name) for name, df in datasets.items()],
        axis=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    sns.histplot(data=combined_df, x=selected_indicator, hue='Dataset', ax=axes[0],
                 element="step", stat="density", common_norm=False)
    axes[0].set_title(f'Histogram of {selected_indicator}')

    # Box Plot
    sns.boxplot(data=combined_df, x='Dataset', y=selected_indicator,  hue='Dataset', ax=axes[1])
    axes[1].set_title(f'Box Plot of {selected_indicator}')

    plt.tight_layout()
    st.pyplot(fig)
