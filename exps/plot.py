import pandas as pd
import wandb
from matplotlib import pyplot as plt
api = wandb.Api()

# runs = api.runs("orangebai/prune_l1")
records = {}
for method in [0, 2, 3, 4, 5]:
    runs = api.runs("orangebai/prune_l1", filters={"config.method": method})
    acc = [run.history(keys=["val/top1"], x_axis="epoch").iloc[-1]['val/top1'] for run in runs]
    spa = [run.history(keys=["sparsity/global"], x_axis="epoch").iloc[-1]['sparsity/global'] for run in runs]
    records[method] = pd.DataFrame({"acc": acc, "spa": spa}).sort_values(by='spa')

for (method, df), marker in zip(records.items(), ['*', 'o', '.', 'x', 'd']):
    plt.plot('spa', 'acc', data=df, marker=marker, label=method)
plt.legend()
plt.show()
