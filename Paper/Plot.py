import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("All.csv")
y_values = df.columns.values

x_axis = y_values[0]
ys = ['Accuracy',"Brier's", "Cohen's", "F1-Macro", "PR-AUC-Trapezoid","ROC-AUC-Trapezoid","PPV","NPV"]
for y in ys:
    print(y)
    plt.plot(df[x_axis], df[y])
    plt.ylabel("Score")
    plt.xlabel("Outcome Prevalence")
plt.legend(ys,prop={'size': 8})
plt.savefig("Plot.pdf")