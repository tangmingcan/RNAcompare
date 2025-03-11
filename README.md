# RNAcompare
Gene expression analysis is crucial for understanding the biological mechanisms underlying patient subgroup differences. However, most existing studies focus primarily on transcriptomic data while neglecting the integration of clinical heterogeneity. Although batch correction methods are commonly used, challenges remain when integrating data across different tissues, omics layers, and diseases. This limitation hampers the ability to connect molecular insights to phenotypic outcomes, thereby restricting clinical translation. Furthermore, the technical complexity of analysing large, heterogeneous datasets poses a barrier for clinicians. To address these challenges, we present RNAcompare, a development of RNAcare, employing machine learning techniques to integrate clinical and multi-omics data seamlessly.

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
#### Step 2: Set Up a Virtual Environment(for development, we use python 3.11)
It's a good practice to use a virtual environment to manage your project's dependencies.
```python
# Install virtualenv if you haven't already
pip install virtualenv

# Create a virtual environment
python3 -m virtualenv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
#### Step 4: Configure the Django Settings
Ensure the Django settings file is configured correctly. The default settings for SQLite should be fine if you're running it locally.

Open the settings.py file in your Django project directory and check the database settings and modify the uploaded folder:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / "db.sqlite3",
    }
}

# change to your own folder for storing user specific uploaded expression and clinic files and storing the shared cohorts data
MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "uploaded")

# change to your local server if it is not 127.0.0.1
ALLOWED_HOSTS = ["127.0.0.1"]
```
You might also want to change the directory for user uploaded files from the default value to some other directory that you have write access to:
```python
MEDIA_ROOT = os.path.join(BASE_DIR, "uploaded")
```
#### Step 5: Run Migrations to configure database
In order to create the database db.sqlite3 and configure the schema, run the following:
```bash
python manage.py migrate
```
#### Step 6: Create a Superuser (if needed)
If you need to create a superuser for accessing the Django admin interface, you can do so with:
```bash
python manage.py createsuperuser
```
#### Step 7: Configure Celery and Redis
The project is built based on Django + Celery + Redis to avoid some issues of fig.save for matplotlib. In order to use Celery and Redis, please have a reference about how to set up it: https://realpython.com/asynchronous-tasks-with-django-and-celery/

Install Redis server:
```bash
# On Linux/WSL2(Windows)
sudo apt install redis
# On macOS
brew install redis
```
To start the Redis server, open another console:
```bash
redis-server --port 8001
```
Then open another console for test:
```bash
redis-cli -p 8001
```
If successfully connected, you should see a cursor with 127.0.0.1:8001>.

Note: Celery settings are defined in djangoproject/settings.py and djangoproject/Celery.py, with the corresponding port number and serialization method for redis.

Open another console to start Celery:
```bash
python3 -m celery -A djangoproject worker -l info
```
#### Step 8: Run the Django Development Server
Finally, run the Django development server to verify that everything is set up correctly.
```bash
python manage.py runserver
```
Open your web browser and go to http://127.0.0.1:8000/ to see your Django project running.

##### 8.1 Upload shared dataset
Similar to RNAcare, you can upload shared dataset into the platform for demonstration. 
##### 8.2 Role management for users
Similar to RNAcare, you can do role management for different users to control their access to shared datasets.

#### Step 9(optional): Nginx setting
Edit your Nginx configuration file (e.g., /etc/nginx/sites-available/your_site.conf) and ensure it includes the following directive inside the appropriate location or server block:
```bash
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Real-IP $remote_addr;
```
Reload Nginx to apply the changes:`sudo systemctl reload nginx`
#### Step 10: Constants settings
In constants.py, there are a few global parameters:
```python
ALLOW_UPLOAD = True # This is to control the platform whether to allow user to upload data
SHAP_PLOT_DATASET='X' # This is for SHAP importance plot and dependence plot based on X/X_train/X_test dataset.
```
## System Introduction
### Platform Features:
```
1. Association Analysis
2. Multi-omics Analysis
3. Imbalanced data consideration
4. Causal Analysis/Inference
5. Enrichment Analysis
```
### Materials and Methods
RNAcompare is a development of RNAcare, developed in Python, utilizing several widely used packages. The webserver was built using the Django framework, adhering to the FAIR (Findable, Accessible, Interoperable, and Reusable) principles. The user interface is constructed with a sidebar containing widgets for data collection and transformation options. The platform employs the Plotly graphics system for generating interactive visualizations.
#### Workflow overview
##### A - Data Upload 
The user has the option to use data uploaded previously, upload their data on the fly or a combination of both. As proof of concept, we included the five datasets described above, including clinical and expression data. RNA-Seq requires a read count matrix, which will be normalised later. Other omics data need to be normalised initially before uploading. Clinical data are uploaded as a table. The aim of normalisation is for association analysis, but for Causal analysis, normalisation is not necessary.
##### B - Data Integration & Visualisation 
For association analysis, data integration & visualization is recommended for finding differences across cohorts. This is applied for situations where batches are from the same tissue and same disease. Integration will help increase the number of samples while removing batch effects.
Similar to RNAcare, RNAcompare detects whether the format of the expression data are integers or non-integers before data integration. For integers, the platform handles the expression data as RNA-Seq data, which are transformed from raw counts to CPM (count per million). For non-integers, for example microarray/proteomic data, the platform handles the expression data as normalised data, so users need to pre-normalize these data types if they want.
The user has the option to log1p transform their data. RNA-Seq data is often highly skewed, with large differences in scale between genes. Log transformation helps stabilise the variance. If this option is selected, the gene expression data will undergo log1p transformation and clinical numeric data will remain the same. Conversely, if the log1p transformation is not selected, the gene expression data will remain unchanged as well. 
Next, the user has two options for the integration, Harmony and Combat, and three options for feature reduction, principal component analysis (PCA), t-distributed stochastic neighbour embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP). In general, it is unclear a priori which integration will generate the best data, and this will vary for different datasets, the user has different options for this in RNAcompare.
After integration, user can view the result based on feature reduction method they selected before association analysis.
##### C – Cohort Comparison
In this section, users have options to choose how to compare different cohorts. We provide 3 different level options for result consistency check. Comparison based on Independent Component Analysis (ICA24) and cohort labels; Comparison based on only Independent Component Analysis; Comparison based on Enrichment Score with Over Representation Analysis(ORA). Their pros and cons will be discussed later in their respective modules. After users choose the option, a new page corresponding to the method will be opened.
Cohorts can have the different definition, depending on the scenarios. In our case studies, it is a categorical label representing the heterogeneous attribute, such as drugs, tissue types or disease types. Other options can be a label standing for 2 different stages of one disease. As we all know DEGs, sensitive to batch correction, is used for finding the different expression genes between the cohorts. Our platform is trying to looking for similarities, where records from different cohorts with strong heterogeneity usually are not easily merged together after UMAP and clustering methods such as K-Means25. 
##### D – Based on ICA and cohort labels
Independent component analysis (ICA) attempts to decompose a multivariate signal into independent non-Gaussian signals. We introduced ICA into multi-omics to decompose the gene expression matrix into independent components. Each independent component was treated as a metagene, characterised by a co-expression pattern and was associated with certain meaningful biological pathway. In practice, we suggest that all clinical fields uploaded before must be named as strings starting with “c_”, such as “c_das”, which will be easy for the program to recognise the expression data and apply ICA algorithm.
In the option, ICA is applied to the cohort labels separately. Say, for ORBIT, we have 2 drugs (anti-TNF and Rituximab) corresponding to 2 different response results: responder and non-responder. Then at first, we need to find the similarities between anti-TNF responders and Rituximab responders. Then we need to find the differences between Rituximab responders and anti-TNF non-responders. In each step, ICA will be applied to the two different states separately and we use the fixed features from Rituximab responders as the bridge, connecting anti-TNF responder and non-responder. The advantage is each disease cohort has its independent ICA components, representing its own state where imbalanced records across different cohorts will not be a problem and during the down streaming analysis, we know where the feature comes from. This is very important because during the analysis we may find some malignant feature, and we can know whether cohort A or B has this malignant feature especially when none of the cohorts is a control group. However, the disadvantage is some of the components across different cohorts may be highly correlated. Therefore, a feature selection method needs to be done after the combination of all components from different cohorts. In our case, we calculate the Pearson correlation matrix26, and use the threshold (0.65 as default) set by users to exclude overlapping features.
Or we can first group the above-mentioned records into 2 cohorts: responders and non-responders before uploading. It all depends on the chosen granularity as long as the number of records in each cohort is enough (at least 10, we recommend).
##### E – Based on ICA
In the option, ICA is applied to all the cohorts at one time. The advantage is that it decreases the possibility of generating overlapping components. The disadvantage is if cohorts are imbalanced, ICA may not capture the specific component, and as mentioned user will not know whether a malignant feature is more closed to cohort A or B if none of them is a control group. For the former issue, Synthetic Minority Oversampling Technique (SMOTE) will be a potential solution via over-sampling the small-sized cohort and under-sampling the large-sized cohort. Another down side is that as the total number of the records processed by ICA increases, its final result may be not stable. To solve this, user needs to increase the number of iterations and number of components to get a relatively stable result. However, it will challenge the time complexity of the online real-time analysis.
##### F – Based on ORA
Over-representation analysis (ORA) is used to determine which a priori defined gene sets are more present (over-represented) in a subset of “interesting” genes than what would be expected by chance28. To infer functional enrichment scores, we will run the ORA method. As input data, it accepts an expression matrix. The top 5% of expressed genes by sample are selected as the set of interest. The final score is obtained by log-transforming the obtained p-values, meaning that higher values are more significant.
We then map the score into pathways according to the annotated gene sets in Molecular Signatures Database (MSigDB). Users can choose different gene sets to compare among cohorts, for example Hallmark or Reactome. In our example, we choose Reactome and cell type signature.
##### G – Association Analysis
Here, the user has the option to associate selected clinical data parameters with the (harmonised) omics data in order to study phenotypes. We provide Random Forests30, XGBoost31 and LightGBM32 with its corresponding parameters for basic tuning. The data will be split into training and test (20%) datasets. AUC-ROC33 or MSE34 will be shown according to different cases. This module will provide SHAP35 feature importance plot and SHAP dependence plot.
SHAP (SHAPley Additive exPlanations) values are a way to explain the output of any machine learning model. It uses a game theoretic approach that measures each player's contribution to the final outcome. In machine learning, each feature is assigned an importance value representing its contribution to the model's output. SHAP values show how each feature affects each final prediction, the significance of each feature compared to others, and the model's reliance on the interaction between features.
##### H – Causal analysis
In the option, users can do causal analysis on treatment efficacy by reducing the selection bias between treatment and control group and handle heterogeneity among patients.
The platform provides Causal Forests based on PCA/ICA processed components for cohorts. Causal Forests is a machine learning method used for estimating heterogeneous treatment effects (HTE) in observational data. It is an extension of Random Forests designed specifically for causal inference. 
The reason why we provide both PCA and ICA processed components is because we think when focusing on HTE, we think PCA will be more robust. However, when focusing on explainable components, ICA will be better.
When using this module, user needs to designate T and Y as the parameters. T is the label representing treatment effect or the predefined cohorts, which we limit only for categorical variables; Y is the dependent variable for phenotype.
##### I – Double Machine Learning
After that we generialised the algorithm to double machine learning (DML), which means two machine learning models to overcome Causal Forests’ limitation.
We then extend Causal Forests and DML to the similarity exploration across different tissues, omics levels and diseases. Under this situation, data harmalisation step is not necessary, which will avoid from potentially removing the biological meaning.
##### J – Enrichment analysis
As the previous analysis will return components of interest, this tab helps to identify biological pathways and signature sets significantly enriched in the dataset.

![image](https://github.com/user-attachments/assets/24f34712-6ffa-4575-9b6e-377a6c12630b)

## Result & Demonstration
![image](https://github.com/user-attachments/assets/1b6d2846-6e3d-42a1-92c5-e87ee6e66bfc)


