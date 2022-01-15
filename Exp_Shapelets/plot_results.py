import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_conf_matrix(array):
    ax = sns.heatmap(array, cmap="cool", annot=True, xticklabels=['Attack', 'Normal'], yticklabels=['Attack', 'Normal'], cbar=False, fmt = ".1f")
    ax.set(title='Confusion Matrix', xlabel="predicted label", ylabel="true label")

    plt.show()


def plot_result():
    df_results = pd.read_csv("TestReports\\Table_Results.csv")
    df_results = df_results[df_results['classifier'] != 'RF'] # Remove RF to many in one plot
    g_1 = df_results.groupby(['window', 'successive_negatives', 'classifier']).agg({ 'F1_score': 'mean'})
    g_1 = g_1.reset_index()

    palette = sns.color_palette("tab10", 3)
    # sns.set(rc={'figure.figsize':(20,8)})
    markers = {25: "s", 50: "X"}
    ax = sns.scatterplot(y="F1_score", x="window", hue="classifier", style='successive_negatives',
                         palette=palette,  data=g_1, markers=markers,  size="classifier", sizes=(50, 120))
    ax.set_xticks(range(100,800,50))
    ax.set_xticklabels(['100','','','250','','','','','500','','','','','750'])

    ax.set(ylabel="f1-score", xlabel='Window size', title='Classifier evaluation')
    plt.show()

plot_result()
plot_conf_matrix([[576, 32], [132, 1133]])
