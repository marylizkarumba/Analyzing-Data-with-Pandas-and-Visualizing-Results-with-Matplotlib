# Data Analysis Project

## Overview
This project demonstrates data loading, exploration, analysis, and visualization using Python's pandas and matplotlib libraries. It processes a dataset (either the Iris dataset or a generated sales dataset), performs basic statistical analysis, and creates visual representations of the data patterns.

## Requirements
- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (for the Iris dataset)

You can install the required packages using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Features

### 1. Data Loading and Exploration
- Loads the Iris dataset (with built-in fallback to a generated sales dataset)
- Displays the first few rows of the dataset
- Examines dataset structure (shape, data types)
- Identifies and handles missing values

### 2. Data Analysis
- Computes basic statistics of numerical columns
- Performs grouping analysis on categorical columns
- Identifies patterns and trends in the data
- Includes time-based analysis for time series data

### 3. Data Visualization
The project creates several types of visualizations:
- **Line chart**: Shows trends over time
- **Bar chart**: Compares numerical values across categories
- **Histogram**: Displays distribution of numerical data
- **Scatter plot**: Visualizes relationships between variables
- **Correlation heatmap**: Shows relationships between all numerical variables

## Project Structure
```
data-analysis-project/
│
├── data_analysis_assignment.py   # Main script
├── README.md                     # This file
│
└── plots/                        # Generated visualizations
    ├── line_chart_time_series.png
    ├── bar_chart_comparison.png
    ├── histogram_distribution.png
    ├── scatter_plot_relationship.png
    └── correlation_heatmap.png
```

## Usage
1. Run the main script:
   ```bash
   python data_analysis_assignment.py
   ```

2. The script will:
   - Load and explore the dataset
   - Perform data analysis
   - Generate visualizations in the 'plots' directory
   - Print analysis results to the console

## Customization
To use your own dataset:
1. Open `data_analysis_assignment.py`
2. In the `load_dataset()` function, uncomment the line:
   ```python
   # df = pd.read_csv('your_dataset.csv')
   ```
3. Replace `'your_dataset.csv'` with the path to your dataset file
4. Run the script as usual

## Converting to Jupyter Notebook
To use this code in a Jupyter notebook:
1. Create a new notebook
2. Copy each function into separate cells
3. Place the main execution code in the final cell
4. Run the cells sequentially



## Error Handling
The script includes robust error handling:
- Graceful handling of missing data
- Fallback dataset generation if loading fails
- Exception handling during visualization creation

## Notes
- The script adaptively chooses appropriate visualizations based on the available column types
- All visualizations are professionally styled with titles, labels, and legends
- Additional information like trend lines and correlation values are included where appropriate