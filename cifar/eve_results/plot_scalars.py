import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

loss_suffix = "-Val-Loss.csv"
acc_suffix  = "-Val-Acc.csv"

best_lrs = [("Adagrad","0.01"), ("Adam","0.001"), ("AMSGrad","0.001"), ("Eve","0.0001"), ("RMSProp","0.001"), ("SGD","0.1")]

legend = []

fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

for rule, lr in best_lrs:
    legend.append(rule)
    loss_filename =  rule + lr + loss_suffix
    frame = pd.read_csv(loss_filename)
    plt.plot(frame['Step'], frame['Value'])

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

ax.legend(legend)

plt.show()

plt.savefig('loss_plot.png')
