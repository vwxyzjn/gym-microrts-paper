import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# sns.set_context("paper", font_scale=1.4)
# plt.rcParams['figure.figsize']=(5,2)
# plt.style.use('science')


name2label = {
    "exp_name: ppo_diverse_impala": "PPO + invalid action masking \n+ diverse bots + IMPALA-CNN",
    "exp_name: ppo_diverse": "PPO + invalid action masking \n + diverse bots",
    "exp_name: ppo_coacai": "PPO + invalid action masking",
    "exp_name: ppo_coacai_naive": "PPO + naive invalid action masking",
    "exp_name: ppo_coacai_partial_mask": "PPO + partial invalid action masking",
    "exp_name: ppo_coacai_no_mask": "PPO",
}

name2label2 = {
    "exp_name: ppo_gridnet_diverse_encode_decode": "PPO + invalid action masking \n+ diverse bots + encoder-decoder",
    "exp_name: ppo_gridnet_diverse_impala": "PPO + invalid action masking \n + diverse bots + IMPALA-CNN",
    "exp_name: ppo_gridnet_diverse": "PPO + invalid action masking \n + diverse bots",
    "exp_name: ppo_gridnet_coacai": "PPO + invalid action masking",
    "exp_name: ppo_gridnet_selfplay_diverse_encode_decode":  "PPO + invalid action masking +\nhalf self-play / half bots + encoder-decoder",
    "exp_name: ppo_gridnet_selfplay_encode_decode":  "PPO + invalid action masking \n+ selfplay + encoder-decoder",
    "exp_name: ppo_gridnet_coacai_naive": "PPO + naive invalid action masking",
    "exp_name: ppo_gridnet_coacai_partial_mask": "PPO + partial invalid action masking",
    "exp_name: ppo_gridnet_coacai_no_mask": "PPO",
}

def read_data(path, name2label):
    data = pd.read_csv(path)
    data['Name'] = data['Name'].map(name2label)
    data = data[data['Name'].notna()]
    data = data.set_index('Name')
    data = data.reindex(list(name2label.values()))
    return data

uas = read_data("uas.csv", name2label)
gridnet = read_data("gridnet.csv", name2label2)
uas_params = read_data("csvs/uas_params.csv", name2label)
gridnet_params = read_data("csvs/gridnet_params.csv", name2label2)
uas_runtime = read_data("csvs/uas_runtime.csv", name2label)
gridnet_runtime = read_data("csvs/gridnet_runtime.csv", name2label2)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("talk")
rs = np.random.RandomState(8)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 3, figsize=(17, 11), sharey='row', sharex='col', gridspec_kw={'height_ratios': [1, 1.6]})
(ax1, ax2, ax3, ax4, ax5, ax6) = axes.flatten()

# sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
palette = sns.color_palette("magma", n_colors=len(gridnet))
# palette.reverse()

bar1 = sns.barplot(
    data=uas,
    y=uas.index,
    x='charts/cumulative_match_results/win rate',
    orient='h',
    capsize=.5,
    ax=ax1,
    palette=palette)
ax1.set_title("")
ax1.set_xlabel("")
ax1.set_ylabel("UAS")
ax1.set_xlim(right=1.2)
for i, v in enumerate(uas['charts/cumulative_match_results/win rate']):
    ax1.text(max(0.05, v +0.05), i+0.2, str(round(v, 2)))

bar2 = sns.barplot(
    data=uas_params,
    y=uas_params.index,
    x='charts/total_parameters',
    orient='h',
    capsize=.5,
    ax=ax2,
    palette=palette)
ax2.set_title("")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_xlim(right=530000)
for i, v in enumerate(uas_params['charts/total_parameters']):
    ax2.text(max(0.05, v +10000), i+0.2, str(round(v/1000000, 2))+"M")

bar3 = sns.barplot(
    y=uas_runtime.index,
    x=uas_runtime['_runtime'] / 3600,
    orient='h',
    capsize=.5,
    ax=ax3,
    palette=palette)
ax3.set_title("")
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_xlim(right=90)
for i, v in enumerate(uas_runtime['_runtime'] / 3600):
    ax3.text(max(0.05, v +2), i+0.2, str(round(v, 2))+"h")

bar4 = sns.barplot(
    data=gridnet,
    y=gridnet.index,
    x='charts/cumulative_match_results/win rate',
    orient='h',
    capsize=.5,
    ax=ax4,
    palette=palette)
ax4.set_title("")
ax4.set_xlabel("Cumulative Win Rate\n(Higher is Better)")
ax4.set_ylabel("Gridnet")
ax4.set_xlim(right=1.2)
for i, v in enumerate(gridnet['charts/cumulative_match_results/win rate']):
    ax4.text(max(0.05, v +0.05), i+0.2, str(round(v, 2)))

bar5 = sns.barplot(
    data=gridnet_params,
    y=gridnet_params.index,
    x='charts/total_parameters',
    orient='h',
    capsize=.5,
    ax=ax5,
    palette=palette)
ax5.set_title("")
ax5.set_xlabel("Parameters in Model")
ax5.set_ylabel("")
ax5.set_xlim(right=7300000)
for i, v in enumerate(gridnet_params['charts/total_parameters']):
    ax5.text(max(0.05, v +10000), i+0.2, str(round(v/1000000, 2))+"M")

bar6 = sns.barplot(
    y=gridnet_runtime.index,
    x=gridnet_runtime['_runtime'] / 3600,
    orient='h',
    capsize=.5,
    ax=ax6,
    palette=palette)
ax6.set_title("")
ax6.set_xlabel("Runtime\n(Lower is Better)")
ax6.set_ylabel("")
ax6.set_xlim(right=270)
for i, v in enumerate(gridnet_runtime['_runtime'] / 3600):
    ax6.text(max(0.05, v +2), i+0.2, str(round(v, 2))+"h")

f.tight_layout()
f.savefig("ablation_all.pdf")
