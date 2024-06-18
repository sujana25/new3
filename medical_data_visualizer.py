import pandas as pd

# 1. Import data
df = pd.read_csv('medical_examination.csv')

#Create the overweight column.
# 2. Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

#Normalize data.
# 3. Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

#Draw the Categorical Plot in the draw_cat_plot function.
import seaborn as sns
import matplotlib.pyplot as plt

def draw_cat_plot():
    # 4. Create DataFrame for cat plot using `pd.melt`
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 5. Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().reset_index(name='total')
    
    # 6. Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar", height=5, aspect=1)
    fig = g.fig
    
    return fig

# Draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 7. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 8. Calculate the correlation matrix
    corr = df_heat.corr()
    
    # 9. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 10. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 11. Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=.5, ax=ax)
    
    return fig

#
