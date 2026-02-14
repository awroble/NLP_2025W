"""Main analysis script for prompts."""

from plots import plot_indicator_over_time, plot_PHtRS_vs_PRtRS
from embeddings import embeddings_analysis
from analysis import analyze_all_prompts

if __name__ == "__main__":

    prompts_analysis = analyze_all_prompts()
    print(prompts_analysis)

    plot_indicator_over_time(prompts_analysis, indicator='S', save_path='./analysis/analysis_plots/s.png')
    plot_indicator_over_time(prompts_analysis, indicator='NCtW', save_path='./analysis/analysis_plots/nctw.png')
    plot_indicator_over_time(prompts_analysis, indicator='CStA', save_path='./analysis/analysis_plots/csta.png')
    plot_PHtRS_vs_PRtRS(prompts_analysis, save_path='./analysis/analysis_plots/phtrs_vs_prtrs.png')                         
    embeddings_analysis('./analysis/analysis_plots/dendrogram.png')