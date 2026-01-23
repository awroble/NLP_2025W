# International Agreements Database Mining

This repository contains deliverables for the NLP course on Warsaw Technology University. The project goal is to solve 13 data mining problems on a database containg large amount of the US international agreements.

## Prerequisites

To reproduce the results, **Python** in version **3.11** or higher is required. For dependency management **Poetry* and `pyproject.toml` file should be utilized. 
### Install Poetry

`curl -sSL https://install.python-poetry.org | python3 -`

### Install dependencies

`poetry install`

## Data

The original data shared with us by the Uniwersytet Łódzki team is stored at [this](https://uniwersytetlodzki-my.sharepoint.com/personal/marcin_frenkel_wsmip_uni_lodz_pl/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmarcin%5Ffrenkel%5Fwsmip%5Funi%5Flodz%5Fpl%2FDocuments%2FU%C5%81%2FProjekt%20z%20Politechnik%C4%85%20Warszawsk%C4%85%20%2D%20baza%20um%C3%B3w%2FBaza%20um%C3%B3w&ga=1) address.

Preprocessed data needed to reproduce the results can be found at [this](https://drive.google.com/drive/folders/12lgLmvTZY6Q3Rz2L9JxGmPggIAXDLJ7M?fbclid=IwY2xjawPIwK5leHRuA2FlbQIxMABicmlkETE4WTE3dzRGVFlRUmY1WDVPc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHikF555zPJ0CW8CP25tOVf89vdwAn_tyd7s-nsh-fzPK306j5YHTIKBhPaGO_aem_REAubR4osJ8cwvnihpEqpA) link.

## Repository structure

**LICENSE**: Defines the legal permissions and restrictions for the code and dataset usage.

**report_source/**: A directory containing **LaTeX** project source code used to render the report.

**preliminary_report.pdf**: A compiled document representing the Milestone 2 report.

**POC_tasks_x_y_z.ipynb**: Jupyter Notebooks containing source code addressing tasks x, y, and z of the assigned project.

**EDA.ipynb**: Jupyter Notebook containing the source code of conducted exploratory data analysis.