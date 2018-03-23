import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

loss_suffix = "-Val-Loss.csv"
acc_suffix  = "-Val-Acc.csv"

best_lrs = [("Adadelta","0.1"), ("Adagrad","0.01"), ("Adam","0.001"),
            ("Adamax","0.001"), ("AMSGrad","0.001"), ("Eve","0.0001"),
            ("RMSProp","0.001"), ("SGD","0.1"), ("SGDMomentum","0.01"),
            ("SGDNesterovMomentum","0.01")]

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

fig.savefig('loss_plot.pdf')


fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

optimizers = ["Adam", "Eve"]
styles    = ['-','--']
layers = [("4-Layers","palevioletred"), ("6-Layers", "lightseagreen"),("8-Layers", "tomato")]

legend = []

for optim, style in zip(optimizers, styles):
    for layer, colour in layers:
        legend.append((optim + layer))
        loss_filename = layer + optim + loss_suffix
        frame = pd.read_csv(loss_filename)
        print(loss_filename, min(frame['Value']))
        plt.plot(frame['Step'][0::10], frame['Value'][0::10], color=colour, linestyle=style)



plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

ax.legend(legend)

fig.savefig('deep_plot.pdf')
