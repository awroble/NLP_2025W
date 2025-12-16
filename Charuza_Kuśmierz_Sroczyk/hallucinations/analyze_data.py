import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
INPUT_FILE = 'dataset_evaluation.json'
OUT_IMG_1 = 'accuracy_by_category.png'
OUT_IMG_2 = 'length_vs_score.png'
PLOT_STYLE = 'whitegrid'
FIG_SIZE = (10, 5)
# =================================================

with open(INPUT_FILE) as f:
    df = pd.json_normalize(json.load(f)['results'])

df['category'] = df['sample_id'].str.split('-').str[0]
df['len'] = df['model_response'].str.len()
df['score'] = df['score'].astype(int)
sns.set_theme(style=PLOT_STYLE)

plt.figure(figsize=FIG_SIZE)
sns.barplot(df, x='category', y='score', palette='viridis', hue='category', legend=False).set(title='Accuracy by Category for the hallucination test', ylabel='Accuracy')
plt.savefig(OUT_IMG_1, bbox_inches='tight')

plt.figure(figsize=FIG_SIZE)
sns.boxplot(df, x='score', y='len', palette='coolwarm', hue='score', legend=False).set(title='Response Length Distribution for the hallucination test')
plt.savefig(OUT_IMG_2, bbox_inches='tight')