# RNAcompare
RNAcompare overcomes these challenges by providing an interactive, reproducible platform designed to analyse multi-omics data in a clinical context. This tool enables researchers to integrate diverse datasets, conduct exploratory analyses, and identify shared patterns across patient subgroups. The platform facilitates hypothesis generation and produces intuitive visualizations to support further investigation.
As a proof of concept, we applied RNAcompare to connect omics data to pain,drug resistance in rheumatoid arthritis (RA) and disease severity to RA and Heart Failure (HF). Our analysis reduced selection bias and managed heterogeneity by identifying key contributors to treatment variability. We discovered shared molecular pathways associated with different treatments. Using SHAP (Shapley Additive Explanations) values, we successfully classified patients into three subgroups based on age, and subsequent analyses confirmed these age-related patterns. Additionally, we uncovered hidden patterns influencing pain and disease severity across different tissues, omics layers, and diseases. Notably, by integrating Causal Forests and Double Machine learning with clinical phenotypes, RNAcompare provides a novel approach to bypass traditional batch correction methods.

## System Implementation and Package dependence
Operating system(s): Platform independent
Compatible bBrowsers: Firefox/Chrome
Programming language: Python, JavaScript
Other requirements: Python >= 3.11, Django >= 4.2, Nginx(optional)
Any restrictions to use by non-academics: licence needed
### 1. Complete code for migration:
To copy RNAcompare from GitHub and make it runnable on your local machine, you can follow these steps:
#### Step 1: Clone the Repository
First, clone the repository from GitHub to your local machine.
```bash
https://github.com/tangmingcan/RNAcompare.git
cd RNAcompare
```
