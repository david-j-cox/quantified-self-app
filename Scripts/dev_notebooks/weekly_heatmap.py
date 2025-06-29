import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_weekly_heatmap(df, title='Exercise Features by Day of Week', figsize=(30, 6), cmap='viridis', exclude_cols=None):
    """
    Create a weekly heatmap of features, with the option to exclude specific columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot
    title : str, optional
        Title for the heatmap
    figsize : tuple, optional
        Figure size as (width, height)
    cmap : str, optional
        Colormap to use for the heatmap
    exclude_cols : list, optional
        List of column names to exclude from the heatmap
    """
    # Create a copy to avoid modifying the original DataFrame
    df_heatmap = df.copy()
    
    # Ensure day_of_week is in the correct format
    if 'day_of_week' not in df_heatmap.columns:
        raise ValueError("DataFrame must contain a 'day_of_week' column")
    
    # Drop excluded columns if specified
    if exclude_cols:
        df_heatmap = df_heatmap.drop(exclude_cols, axis=1)
    
    # Group by day_of_week and aggregate (mean)
    df_grouped = df_heatmap.groupby('day_of_week').mean().round(2)
    
    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_grouped = df_grouped.reindex(days_order)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df_grouped, annot=True, cmap=cmap)
    plt.title(title)
    plt.ylabel('Day of Week', fontsize=30, labelpad=12)
    plt.xlabel('')
    plt.show() 