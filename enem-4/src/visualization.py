from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Visualization:

    @staticmethod
    def correlation_features(df):
        
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

    @staticmethod
    def verify_outliers(df, min, max, columns):
        df[columns].boxplot()

        for column in columns:
            min_outlier = df.query(f"{column} < {min}")
            df.drop(min_outlier.index, inplace=True)

            max_outlier = df.query(f"{column} > {max}")
            df.drop(max_outlier.index, inplace=True)

        return df

    @staticmethod
    def balancing_analysis(df, target):
        pca = PCA(n_components=2)

        pca.fit(df)

        imbalanced_pca = pca.transform(df)
        sns.scatterplot(imbalanced_pca[:, 0], imbalanced_pca[:, 1], hue=target)


    