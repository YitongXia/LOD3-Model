import json
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Palatino'
mpl.rcParams['mathtext.fontset'] = 'cm'

import json
import matplotlib.pyplot as plt

# # Read the metrics.json file
# with open('metrics.json', 'r') as f:
#     metrics = json.load(f)
#
# # Extract loss and accuracy data
# loss = metrics['loss']
# accuracy = metrics['accuracy']
#
# # Create a common x-axis for both loss and accuracy
# x_axis = list(range(1, len(loss) + 1))
#
# # Plot the loss and accuracy on the same x-y axis
# plt.plot(x_axis, loss, label='Loss', color='tab:red')
# plt.plot(x_axis, accuracy, label='Accuracy', color='tab:blue')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss / Accuracy')
# plt.title('Loss and Accuracy in a Single Figure')
# plt.legend()
#
# plt.show()

# Read metrics.json line by line
metrics_data = []
with open("../../import_Almere/metrics_5k.json", "r") as f:
    for line in f:
        metrics_data.append(json.loads(line))

# Extract data for specific metrics
iterations = [metric['iteration'] for metric in metrics_data]
losses = [metric['total_loss'] for metric in metrics_data]
accuracy = [metric['mask_rcnn/accuracy'] for metric in metrics_data]
false_neg = [metric['mask_rcnn/false_negative'] for metric in metrics_data]

# Plot the loss and accuracy on the same x-y axis
plt.plot(iterations, losses, label='Total Loss', color='tab:red')
plt.plot(iterations, accuracy, label='Accuracy', color='tab:blue')
plt.plot(iterations, false_neg, label='False Negative', color='yellow')

plt.xlabel('Iteration')
plt.ylabel('Loss / Accuracy / False Negative')
plt.legend()
plt.title('iteration time: 5,000  backbone: ResNet-101')
# Plot total_loss
plt.figure()
plt.plot(iterations, losses)
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.title("Total Loss vs Iteration")
plt.show()

# Plot AP50
plt.figure()
plt.plot(iterations, accuracy)
plt.xlabel("Iteration")
plt.ylabel("AP50")
plt.title("AP50 vs Iteration")
plt.show()