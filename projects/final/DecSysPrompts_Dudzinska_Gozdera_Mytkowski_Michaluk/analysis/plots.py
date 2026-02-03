"""Plotting functions for prompts analysis."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import ternary
from scipy.stats import zscore

def plot_indicator_over_time(prompts_analysis, indicator = 'S', save_path = None):
    """Plot the evolution of a given prompt indicator over time, differentiated by provider."""

    prompts_analysis = prompts_analysis.sort_values('release_date')

    fig, ax = plt.subplots(figsize=(10, 4))

    providers = prompts_analysis['provider'].unique()
    colors = ['orange', 'red', 'black', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    provider_colors = {}
    for i, provider in enumerate(providers):
        provider_colors[provider] = colors[i % len(colors)]

    for provider in providers:
        provider_data = prompts_analysis[prompts_analysis['provider'] == provider]

        ax.scatter(
            provider_data['release_date'], 
            provider_data[indicator],
            color=provider_colors[provider],
            label=provider,
            s=50
        )

        ax.plot(
            provider_data['release_date'],
            provider_data[indicator],
            color=provider_colors[provider],
            linewidth=2
        )

    ax.set_xlabel('Release Date', fontsize=24)
    ax.set_ylabel(f'Prompt {indicator}', fontsize=24)
    #ax.set_title(f'Prompt {indicator} Over Time', fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    ax.legend(fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_PHtRS_vs_PRtRS(prompts_analysis, save_path=None):
    """Plot PHtRS against PRtRS for all prompts, differentiated by provider."""

    fig, ax = plt.subplots(figsize=(10, 4))

    providers = prompts_analysis['provider'].unique()
    colors = ['red', 'blue', 'black', 'orange', 'green', 'brown', 'pink', 'gray', 'olive', 'cyan']
    provider_colors = {}
    for i, provider in enumerate(providers):
        provider_colors[provider] = colors[i % len(colors)]

    for provider in providers:
        provider_data = prompts_analysis[prompts_analysis['provider'] == provider]

        ax.scatter(
            provider_data['PRtRS'], 
            provider_data['PHtRS'],
            color=provider_colors[provider],
            label=provider,
            s=200
        )

    ax.set_xlabel('PRtRS', fontsize=24)
    ax.set_ylabel('PHtRS', fontsize=24)
    #ax.set_title('PHtRS vs PRtRS', fontsize=16)

    ax.legend(fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_scatter_grouped_by_provider(
    df,
    x_col='FRE',
    y_col='Size (Characters)'
):
    # 1. Group by provider
    grouped = (
        df
        .groupby('provider', as_index=False)
        .agg({
            x_col: 'mean',
            y_col: 'mean'
        })
    )

    # 2. Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=grouped,
        x=x_col,
        y=y_col,
        hue='provider',
        style='provider',
        s=300,  # increase point size
        ax=ax
    )

    # 3. Formatting with bold and bigger fonts
    ax.set_title('Prompt Complexity by Provider (Averaged)', fontsize=18, fontweight='bold')
    ax.set_xlabel(f'{x_col} (0–100)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{y_col} (Number of Characters)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 4. Legend formatting
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title='Provider',
        title_fontsize=12,
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


def plot_scatter_with_outliers(df, x_col='FRE', y_col='Size (Characters)', z_threshold = 2.0):
    # 1. Calculate Z-Scores
    df_copy = df.copy()
    df_copy['z_fre'] = zscore(df_copy[x_col])
    df_copy['z_size'] = zscore(df_copy[y_col])

    # 2. Setup the plot
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df_copy,
        x=x_col, 
        y=y_col, 
        hue='provider',    
        style='provider',  
        s=100              
    )

    # 3. Calculate the actual values for Mean and 2.0 Std Dev
    mean_fre = df_copy[x_col].mean()
    std_fre = df_copy[x_col].std()
    mean_size = df_copy[y_col].mean()
    std_size = df_copy[y_col].std()

    # 4. Draw the 2.0 Std Dev Lines (The "Outlier Boundaries")
    plt.axvline(x=mean_fre + 2 * std_fre, color='green', linestyle='--', alpha=0.7, label=f'{x_col} (+{z_threshold} SD)')
    plt.axvline(x=mean_fre - 2 * std_fre, color='green', linestyle='--', alpha=0.7, label=f'{x_col} (-{z_threshold} SD)')
    plt.axhline(y=mean_size + 2 * std_size, color='orange', linestyle='--', alpha=0.7, label=f'{y_col} (+{z_threshold} SD)')

    # 5. Label the Outliers with Z-Score Threshold
    for i, row in df_copy.iterrows():
        if abs(row['z_fre']) > z_threshold or abs(row['z_size']) > z_threshold:
            label = f"{row['model']} {row['version']}" if 'version' in df_copy.columns else row['model']
            plt.text(
                x=row[x_col] + 1, 
                y=row[y_col], 
                s=label, 
                fontsize=9,
                color='red',
                weight='bold'
            )

    plt.title(f'Prompt Complexity with Outlier Boundaries ({z_threshold} SD)', fontsize=14)
    plt.xlabel(f'{x_col} (0-100)', fontsize=12)
    plt.ylabel(f'{y_col} (Number of Characters)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 

    plt.show()

def plot_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include='number').columns
    non_zero_cols = numeric_cols[(df[numeric_cols] != 0).any()]
    print(non_zero_cols)
    non_zero_df = df.loc[:, non_zero_cols]
    corr_matrix = non_zero_df.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
def plot_regulative_statements_heatmap(prompts_analysis):
    df_grouped = (
    prompts_analysis
    .groupby("provider")[["StRS", "CtRS", "PRtRS", "PHtRS"]]
    .mean()
    .reset_index()
    )
    df_indexed = df_grouped.set_index("provider")
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        df_indexed,
        annot=True,
        cmap='coolwarm',
        fmt=".3f",
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.show()
    
def plot_ternary_plot(prompts_analysis):
    df_grouped = (
    prompts_analysis
    .groupby("provider")[["StRS", "CtRS", "PRtRS", "PHtRS"]]
    .mean()
    .reset_index()
    )
    df_metrics = df_grouped.drop(columns=["StRS"])
    df_metrics.set_index("provider", inplace=True)
    df_normalized = df_metrics.div(df_metrics.sum(axis=1), axis=0) * 100
    
    three_way = df_normalized[['CtRS', 'PRtRS', 'PHtRS']].copy()
    three_way_pct = three_way.div(three_way.sum(axis=1), axis=0) * 100
    points = []
    labels = []
    
    for provider, row in three_way_pct.iterrows():
        # Each point is (command%, prohibition%, permission%)
        point = (row['CtRS'], row['PHtRS'], row['PRtRS'])
        points.append(point)
        labels.append(provider)

    ### Scatter Plot
    scale = 100  # Use 100 for percentage scale
    figure, tax = ternary.figure(scale=scale)
    # tax.set_title("Governance Philosophy Triangle", fontsize=14, fontweight='bold')
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="gray", alpha=0.4)

    # Plot each provider with different colors
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', "#0a0a0a"]
    for i, (point, label) in enumerate(zip(points, labels)):
        tax.scatter([point], marker='o', s=200, color=colors[i % len(colors)], 
                    label=label, alpha=0.8, edgecolors='black', linewidth=1)

    # Add axis labels
    tax.left_axis_label("Permission % (PRtRS)", fontsize=12, offset=0.14)
    tax.right_axis_label("Prohibition % (PHtRS)", fontsize=12, offset=0.14)
    tax.bottom_axis_label("Command % (CtRS)", fontsize=12, offset=0.08)
    # Add legend and ticks
    tax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    tax.ticks(axis='lbr', linewidth=1, multiple=10, offset=0.02)
    tax.clear_matplotlib_ticks()

    ax = tax.get_axes()
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.patch.set_visible(False)
    tax.show()

    print("\nTernary Plot Data (Command%, Prohibition%, Permission%):")
