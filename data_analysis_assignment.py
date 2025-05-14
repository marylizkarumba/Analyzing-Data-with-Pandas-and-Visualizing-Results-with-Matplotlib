#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Analysis Assignment
------------------------
This script demonstrates data loading, exploration, analysis, and visualization
using pandas, matplotlib, and seaborn to fulfill the assignment requirements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os
from datetime import datetime, timedelta

# Set plot style for more appealing visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create directory for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# =============================
# Task 1: Load and Explore the Dataset
# =============================

def load_dataset():
    """
    Load a dataset for analysis. 
    This function will try to load the iris dataset from sklearn as a backup.
    """
    try:
        # Try to load a CSV file - replace with your own dataset path
        # df = pd.read_csv('your_dataset.csv')
        # print("Successfully loaded dataset from CSV file")
        
        # For this example, we'll use the iris dataset
        print("Loading the Iris dataset...")
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Series([iris.target_names[i] for i in iris.target])
        
        # Let's add some missing values for demonstration purposes
        df.loc[np.random.choice(df.index, 10), 'sepal length (cm)'] = np.nan
        df.loc[np.random.choice(df.index, 5), 'petal width (cm)'] = np.nan
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating a sample sales dataset instead...")
        
        # If loading fails, create a sample sales dataset
        np.random.seed(42)
        
        # Create dates
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        # Create product categories
        products = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports']
        
        # Create regions
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        # Generate random sales data
        n_records = 1000
        
        data = {
            'date': np.random.choice(dates, n_records),
            'product': np.random.choice(products, n_records),
            'region': np.random.choice(regions, n_records),
            'units_sold': np.random.randint(1, 50, n_records),
            'unit_price': np.round(np.random.uniform(10, 1000, n_records), 2),
        }
        
        # Calculate revenue
        sales_df = pd.DataFrame(data)
        sales_df['revenue'] = sales_df['units_sold'] * sales_df['unit_price']
        
        # Add some missing values
        sales_df.loc[np.random.choice(sales_df.index, 20), 'units_sold'] = np.nan
        sales_df.loc[np.random.choice(sales_df.index, 15), 'unit_price'] = np.nan
        
        # Convert date to datetime
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        return sales_df

def explore_dataset(df):
    """
    Explore the structure and content of the dataset.
    """
    print("\n===== DATA EXPLORATION =====")
    
    print("\n1. First few rows of the dataset:")
    print(df.head())
    
    print("\n2. Dataset shape (rows, columns):")
    print(df.shape)
    
    print("\n3. Column data types:")
    print(df.dtypes)
    
    print("\n4. Checking for missing values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    print(f"\nTotal missing values: {missing_values.sum()}")
    
    return missing_values

def clean_dataset(df, missing_values):
    """
    Clean the dataset by handling missing values.
    """
    print("\n===== DATA CLEANING =====")
    
    if missing_values.sum() > 0:
        print("Handling missing values...")
        
        # Make a copy of the dataframe to avoid modifying the original
        df_cleaned = df.copy()
        
        # For each column with missing values, either fill or drop
        for column in df.columns:
            missing_count = missing_values[column]
            
            if missing_count > 0:
                print(f"- Column '{column}' has {missing_count} missing values")
                
                # Strategy depends on column type
                if pd.api.types.is_numeric_dtype(df[column]):
                    # For numerical columns, fill with median
                    median_value = df[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)
                    print(f"  Filled with median value: {median_value:.2f}")
                elif pd.api.types.is_datetime64_dtype(df[column]):
                    # For datetime columns, fill with the most recent previous date
                    df_cleaned[column].fillna(method='ffill', inplace=True)
                    print("  Filled with forward fill method")
                else:
                    # For categorical/object columns, fill with mode
                    mode_value = df[column].mode()[0]
                    df_cleaned[column].fillna(mode_value, inplace=True)
                    print(f"  Filled with mode value: '{mode_value}'")
        
        # Verify all missing values have been handled
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"\nRemaining missing values after cleaning: {remaining_missing}")
        
        return df_cleaned
    else:
        print("No missing values to clean.")
        return df

# =============================
# Task 2: Basic Data Analysis
# =============================

def analyze_dataset(df):
    """
    Perform basic statistical analysis on the dataset.
    """
    print("\n===== BASIC DATA ANALYSIS =====")
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle datetime columns separately
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    print("\n1. Basic statistics for numerical columns:")
    if numerical_cols:
        print(df[numerical_cols].describe())
    else:
        print("No numerical columns found.")
    
    print("\n2. Group analysis by categorical columns:")
    
    # Analyze based on categorical columns and compute means of numerical columns
    findings = []
    
    for cat_col in categorical_cols:
        print(f"\nGrouping by: {cat_col}")
        
        # For each numerical column, compute group means
        group_means = df.groupby(cat_col)[numerical_cols].mean().round(2)
        print(group_means)
        
        # Find the highest and lowest values for each numerical column
        for num_col in numerical_cols:
            max_group = group_means[num_col].idxmax()
            min_group = group_means[num_col].idxmin()
            max_val = group_means[num_col].max()
            min_val = group_means[num_col].min()
            
            finding = f"- The highest average {num_col} is in {max_group} ({max_val:.2f})"
            findings.append(finding)
            finding = f"- The lowest average {num_col} is in {min_group} ({min_val:.2f})"
            findings.append(finding)
    
    # If we have datetime columns, analyze time trends
    if datetime_cols:
        print("\n3. Time-based analysis:")
        date_col = datetime_cols[0]  # Use the first datetime column
        
        # Group by month and compute averages
        df['month'] = df[date_col].dt.month
        monthly_avg = df.groupby('month')[numerical_cols].mean().round(2)
        print("\nMonthly averages:")
        print(monthly_avg)
        
        # Find growth trends
        first_month = monthly_avg.iloc[0]
        last_month = monthly_avg.iloc[-1]
        
        for col in numerical_cols:
            if first_month[col] > 0:  # Avoid division by zero
                growth = (last_month[col] - first_month[col]) / first_month[col] * 100
                finding = f"- {col} {'increased' if growth > 0 else 'decreased'} by {abs(growth):.1f}% from first to last month"
                findings.append(finding)
    
    print("\n4. Key findings from analysis:")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    
    return findings

# =============================
# Task 3: Data Visualization
# =============================

def create_visualizations(df):
    """
    Create various visualizations as required by the assignment.
    """
    print("\n===== DATA VISUALIZATION =====")
    
    # Identify column types for appropriate visualizations
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 1. Line chart (time series)
    print("\n1. Creating line chart for time series data...")
    
    try:
        if datetime_cols:
            # If we have datetime columns, create a proper time series
            date_col = datetime_cols[0]
            plt.figure(figsize=(12, 6))
            
            # Group by date and get mean of a numerical column
            num_col = numerical_cols[0]  # Use the first numerical column
            time_series = df.groupby(pd.Grouper(key=date_col, freq='M'))[num_col].mean()
            
            plt.plot(time_series.index, time_series.values, marker='o', linestyle='-', color='#3498db', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Monthly Average {num_col} Over Time', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(num_col, fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig('plots/line_chart_time_series.png')
            plt.close()
            print("  Line chart saved as 'plots/line_chart_time_series.png'")
        else:
            # If no datetime column, create a line chart with numerical indices
            plt.figure(figsize=(12, 6))
            
            # Choose a numerical column
            num_col = numerical_cols[0]
            
            # Get moving average to show a trend
            window_size = 10
            moving_avg = df[num_col].rolling(window=window_size).mean()
            
            plt.plot(df.index, df[num_col], color='#3498db', alpha=0.3, label='Raw data')
            plt.plot(moving_avg.index, moving_avg.values, color='#e74c3c', linewidth=2, label=f'{window_size}-point moving average')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Trend Analysis of {num_col}', fontsize=16)
            plt.xlabel('Index', fontsize=12)
            plt.ylabel(num_col, fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig('plots/line_chart_trend.png')
            plt.close()
            print("  Line chart saved as 'plots/line_chart_trend.png'")
    
    except Exception as e:
        print(f"  Error creating line chart: {e}")
    
    # 2. Bar chart
    print("\n2. Creating bar chart for categorical comparison...")
    
    try:
        if categorical_cols and numerical_cols:
            plt.figure(figsize=(12, 6))
            
            # Choose a categorical and a numerical column
            cat_col = categorical_cols[0]
            num_col = numerical_cols[0]
            
            # Group by the categorical column and compute mean
            grouped_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            
            # Create bar chart
            ax = sns.barplot(x=grouped_data.index, y=grouped_data.values, palette='viridis')
            
            # Add value labels on top of each bar
            for i, v in enumerate(grouped_data.values):
                ax.text(i, v + 0.01 * max(grouped_data.values), f'{v:.2f}', 
                        ha='center', va='bottom', fontsize=10)
            
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.title(f'Average {num_col} by {cat_col}', fontsize=16)
            plt.xlabel(cat_col, fontsize=12)
            plt.ylabel(f'Average {num_col}', fontsize=12)
            plt.xticks(rotation=45 if len(grouped_data) > 5 else 0)
            plt.tight_layout()
            
            plt.savefig('plots/bar_chart_comparison.png')
            plt.close()
            print("  Bar chart saved as 'plots/bar_chart_comparison.png'")
        else:
            print("  Cannot create bar chart: Missing categorical or numerical columns")
    
    except Exception as e:
        print(f"  Error creating bar chart: {e}")
    
    # 3. Histogram
    print("\n3. Creating histogram for distribution analysis...")
    
    try:
        if numerical_cols:
            plt.figure(figsize=(12, 6))
            
            # Choose a numerical column
            num_col = numerical_cols[0]
            
            # Create histogram with KDE
            sns.histplot(df[num_col], kde=True, bins=20, color='#9b59b6')
            
            # Add vertical line for mean and median
            mean_val = df[num_col].mean()
            median_val = df[num_col].median()
            
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Distribution of {num_col}', fontsize=16)
            plt.xlabel(num_col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig('plots/histogram_distribution.png')
            plt.close()
            print("  Histogram saved as 'plots/histogram_distribution.png'")
        else:
            print("  Cannot create histogram: No numerical columns")
    
    except Exception as e:
        print(f"  Error creating histogram: {e}")
    
    # 4. Scatter plot
    print("\n4. Creating scatter plot for relationship analysis...")
    
    try:
        if len(numerical_cols) >= 2:
            plt.figure(figsize=(10, 8))
            
            # Choose two numerical columns
            x_col = numerical_cols[0]
            y_col = numerical_cols[1]
            
            # Add color by categorical column if available
            if categorical_cols:
                hue_col = categorical_cols[0]
                scatter = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                                          palette='viridis', s=70, alpha=0.7)
                plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(data=df, x=x_col, y=y_col, color='#2ecc71', s=70, alpha=0.7)
            
            # Add a trend line
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False, 
                       line_kws={'color': 'red', 'linewidth': 2})
            
            # Calculate correlation
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            plt.annotate(f'Correlation: {corr:.2f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Relationship between {x_col} and {y_col}', fontsize=16)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.tight_layout()
            
            plt.savefig('plots/scatter_plot_relationship.png')
            plt.close()
            print("  Scatter plot saved as 'plots/scatter_plot_relationship.png'")
        else:
            print("  Cannot create scatter plot: Need at least 2 numerical columns")
    
    except Exception as e:
        print(f"  Error creating scatter plot: {e}")
    
    # 5. Bonus: Correlation heatmap for all numerical variables
    print("\n5. Creating correlation heatmap as bonus visualization...")
    
    try:
        if len(numerical_cols) > 1:
            plt.figure(figsize=(10, 8))
            
            # Compute correlation matrix
            corr_matrix = df[numerical_cols].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            
            plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
            plt.tight_layout()
            
            plt.savefig('plots/correlation_heatmap.png')
            plt.close()
            print("  Correlation heatmap saved as 'plots/correlation_heatmap.png'")
        else:
            print("  Cannot create correlation heatmap: Need multiple numerical columns")
    
    except Exception as e:
        print(f"  Error creating correlation heatmap: {e}")
    
    print("\nAll visualizations have been created and saved to the 'plots' directory.")

# =============================
# Main Execution
# =============================

def main():
    """
    Execute the complete data analysis workflow.
    """
    print("Starting data analysis assignment...\n")
    
    try:
        # Task 1: Load and explore the dataset
        df = load_dataset()
        missing_values = explore_dataset(df)
        df_cleaned = clean_dataset(df, missing_values)
        
        # Task 2: Basic data analysis
        findings = analyze_dataset(df_cleaned)
        
        # Task 3: Data visualization
        create_visualizations(df_cleaned)
        
        print("\n===== SUMMARY OF ANALYSIS =====")
        print("\nData loading and exploration:")
        print(f"- Dataset shape: {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns")
        print(f"- Originally had {missing_values.sum()} missing values, now cleaned")
        
        print("\nKey findings from the analysis:")
        for i, finding in enumerate(findings[:5], 1):  # Show top 5 findings
            print(f"{i}. {finding}")
        
        print("\nVisualizations created:")
        print("1. Line chart showing trends over time")
        print("2. Bar chart comparing a numerical value across categories")
        print("3. Histogram showing the distribution of a numerical column")
        print("4. Scatter plot showing the relationship between two numerical columns")
        print("5. Correlation heatmap as a bonus visualization")
        
        print("\nData analysis assignment completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()