from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Visualization:

    def correlation_features(self, df):
        sns.set(style="white")

        # Compute the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})


    def verify_outliers(self, df, min, max):
        
        for column in df.columns:

            min_outlier = df.query(f"{column} < {min}") 
            df.drop(min_outlier.index, inplace = True)

            max_outlier = df.query(f"{column} > {max}")
            df.drop(max_outlier.index, inplace = True)

        return df
    
    