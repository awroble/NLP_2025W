import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
INPUT_FILE = 'gpt-5-mini-wikipedia_dataset2_evaluation.json'
OUT_IMG_1 = 'gpt-5-mini_wiki2_accuracy_by_category.png'
OUT_IMG_2 = 'gpt-5-mini_wiki2_length_vs_score.png'

'''
INPUT_FILE = 'gpt-5-wikipedia_dataset2_evaluation.json'
OUT_IMG_1 = 'gpt-5_wiki2_accuracy_by_category.png'
OUT_IMG_2 = 'gpt-5_wiki2_length_vs_score.png'

INPUT_FILE = 'llama_evaluation.json'
OUT_IMG_1 = 'llama_wiki2_accuracy_by_category.png'
OUT_IMG_2 = 'llama_wiki2_length_vs_score.png'

INPUT_FILE = 'mistral_evaluation.json'
OUT_IMG_1 = 'mistral_wiki2_accuracy_by_category.png'
OUT_IMG_2 = 'mistral_wiki2_length_vs_score.png'

INPUT_FILE = 'llava_evaluation.json'
OUT_IMG_1 = 'llava_wiki2_accuracy_by_category.png'
OUT_IMG_2 = 'llava_wiki2_length_vs_score.png'
'''

PLOT_STYLE = 'whitegrid'
FIG_SIZE = (12, 6)
# =================================================

with open(INPUT_FILE) as f:
    df = pd.json_normalize(json.load(f)['results'])

df['category'] = df['sample_id'].str.split('-').str[1]
df['len'] = df['model_response'].str.len()

df.dropna(subset=['score'], inplace=True)
df['score'] = df['score'].astype(int)

score_mapping = {1: 'Correct', 0: 'Incorrect', 2: 'Refusal'}
df['outcome'] = df['score'].map(score_mapping)

sns.set_theme(style=PLOT_STYLE)

plt.figure(figsize=FIG_SIZE)
proportions = df.groupby('category')['outcome'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
pivot_df = proportions.pivot(index='category', columns='outcome', values='percentage').fillna(0)

outcome_order = ['Correct', 'Incorrect', 'Refusal']
pivot_df = pivot_df.reindex(columns=outcome_order, fill_value=0)

pivot_df.plot(
    kind='bar',
    stacked=True,
    color={'Correct': 'green', 'Incorrect': 'red', 'Refusal': 'orange'},
    figsize=FIG_SIZE,
    width=0.8
)

plt.title('Response Outcomes by Category', fontsize=16)
plt.ylabel('Percentage (%)')
plt.xlabel('Category')
plt.xticks(rotation=0)
plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_IMG_1, bbox_inches='tight')


plt.figure(figsize=FIG_SIZE)
sns.boxplot(data=df, x='outcome', y='len',
            palette={'Correct': 'green', 'Incorrect': 'red', 'Refusal': 'orange'},
            hue='outcome', legend=False, order=['Correct', 'Incorrect', 'Refusal'])
            
plt.title('Response Length Distribution by Outcome', fontsize=16)
plt.ylabel('Response Length (characters)')
plt.xlabel('Outcome')
# plt.yscale('log')
plt.savefig(OUT_IMG_2, bbox_inches='tight')

print(f"Analysis complete. Plots saved to '{OUT_IMG_1}' and '{OUT_IMG_2}'.")