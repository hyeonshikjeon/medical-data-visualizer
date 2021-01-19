import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight']/((df['height']/100)**2) > 25) * 1

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].astype('int32')
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].astype('int32')
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df,id_vars=['cardio'],value_vars=['active','alco','cholesterol','gluc','overweight','smoke'], var_name='variable',value_name='value')


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', hue='value', col='cardio', kind='count', data = df_cat)
    ax = fig.axes.flat[0]
    ax.set_ylabel('total')

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.drop(df[df['ap_lo'] > df['ap_hi']].index)
    df_heat = df_heat.drop(df_heat[df_heat['height'] < df['height'].quantile(0.025)].index)
    df_heat = df_heat.drop(df_heat[df_heat['height'] > df['height'].quantile(0.975)].index)
    df_heat = df_heat.drop(df_heat[df_heat['weight'] < df['weight'].quantile(0.025)].index)
    df_heat = df_heat.drop(df_heat[df_heat['weight'] > df['weight'].quantile(0.975)].index)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True


    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(15,10))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr,mask=mask,linewidth=0.5,vmin=-0.13,vmax=0.3,fmt='.1f',cbar_kws={'shrink':.5,'ticks':[-0.08,0.00,0.08,0.16,0.24]},annot = True, square=True)
    
    fig = ax.get_figure()
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
