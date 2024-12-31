import matplotlib.pyplot as plt
import numpy as np

# 数据
data = {
    '1_freerider': {
        'scheme_A': [0.85, 0.82, 0.88, 0.80],
        'scheme_B': [0.87, 0.84, 0.90, 0.82]
    },
    '5_freerider': {
        'scheme_A': [0.80, 0.78, 0.82, 0.75],
        'scheme_B': [0.83, 0.80, 0.85, 0.78]
    }
}

models = ['a', 'b', 'c', 'd']
freerider_conditions = ['1_freerider', '5_freerider']
schemes = ['scheme_A', 'scheme_B']

# 创建图表
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

bar_width = 0.35
index = np.arange(len(models))

for ax, (model_index, model) in zip(axs, enumerate(models)):
    for i, condition in enumerate(freerider_conditions):
        scheme_A_values = data[condition]['scheme_A']
        scheme_B_values = data[condition]['scheme_B']
        
        ax.bar(index + (i * bar_width), scheme_A_values, bar_width, label=f'Scheme A ({condition})' if model_index == 0 else "")
        ax.bar(index + (i * bar_width) + bar_width, scheme_B_values, bar_width, label=f'Scheme B ({condition})' if model_index == 0 else "")

    ax.set_title(f'Model {model}')
    ax.set_xlabel('Freerider Condition')
    ax.set_ylabel('Top1-Accuracy')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.legend()

plt.tight_layout()
plt.show()
