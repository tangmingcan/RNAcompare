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
**ALLOW_UPLOAD = True # This is to control the platform whether to allow user to upload data**
**SHAP_PLOT_DATASET='X' # This is for SHAP importance plot and dependence plot based on X/X_train/X_test dataset.**

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
### Data Collection
Our platform supports the integration of multi-omics data with clinical data, facilitating a comprehensive analysis that combines not only genetic, transcriptomic, proteomic, and metabolomic data but also RNA-seq and microarray with patient-specific clinical information. This integrated approach allows users to uncover more complex and clinically relevant insights. After clicking the ‘Upload’ button in the sidebar, uploaded information about the study and experiment is displayed in the panel. It shows the first 5 rows for dataset with column names. User will have option to select log1p to transform the dataset later. For later calculation.
User can upload more than one expression files which will be treated as different batches.

![image](https://github.com/user-attachments/assets/1b6d2846-6e3d-42a1-92c5-e87ee6e66bfc)

### Data Integration & Visualization
Similar to RNAcare, user have option to choose built-in compiled cohorts for integration. RNAcompare provide 3 options for integration: Combat, Harmony or No batch correction. When finding the similarity of different batches without using association analysis, we recommend no batch correction method.

![image](https://github.com/user-attachments/assets/b0b8ac87-8890-43fd-9c19-063fe4f79f28)

After uploading the data to the platform, user needs to tell the platform which fields in the clinic data will be treated as LABEL/Batches/cohorts. First, a field called "LABEL" is compulsory for the system to recognize. As you see in the picture, user tells the platform to recognize the other two fields as well, namely 'LABEL1', 'LABEL2'. 

**LABEL is the label for disguishing patients for drug response or non-response group.**

**LABEL1 is the label for disguishing patients for anti-TNF response/non-response or Rituximab response/non-response group.**

**LABEL2 is the label for disguishing patients for anti-TNF or Rituximab.**

After data processing, user will be guided to data visualization part to overview the data.

![image](https://github.com/user-attachments/assets/8ebc0433-38cb-44ec-92f7-006a12555f70)

### Cohort Comparision
In this section, users have options to choose how to compare different cohorts. We provide 3 different level options for result consistency check. Comparison based on ICA and cohort labels; Comparison based on only Independent Component Analysis; Comparison based on Enrichment Score with Over Representation Analysis(ORA). Their pros and cons are discussed in the paper. After users choose the option, a new page corresponding to the method will be opened.

Cohorts can have the different definition, depending on the scenarios. In our case studies, it is a categorical label representing the heterogeneous attribute, such as drugs, tissue types or disease types. Other options can be a label standing for 2 different stages of one disease. As we all know DEGs, sensitive to batch correction, is used for finding the different expression genes between the cohorts. Our platform is trying to looking for similarities, where records from different cohorts with strong heterogeneity usually are not easily merged together after UMAP and clustering methods such as K-Means.

![image](https://github.com/user-attachments/assets/e9879697-c017-48d1-9ca9-61c43c1b5030)

### Based on ICA and cohort labels
In the option, ICA is applied to the cohort labels separately. This is very important because during the analysis we may find some malignant feature, and we can know whether cohort A or B has this malignant feature especially when none of the cohorts is a control group. However, the disadvantage is some of the components across different cohorts may be highly correlated. Therefore, a feature selection method needs to be done after the combination of all components from different cohorts. In our case, we calculate the Pearson correlation matrix, and use the threshold (0.65 as default) set later in ML algorithm settings by users to exclude overlapping features.

![image](https://github.com/user-attachments/assets/928015f9-8c0d-41db-8636-b1a6544c5e18)


### Based on ICA
In the option, ICA is applied to all the cohorts at one time. The advantage is that it decreases the possibility of generating overlapping components. The disadvantage is if cohorts are imbalanced, ICA may not capture the specific component, and as mentioned user will not know whether a malignant feature is more closed to cohort A or B if none of them is a control group.

![image](https://github.com/user-attachments/assets/b914b3c8-b2ba-4271-8e4d-91acef825560)

At the same time, user have option to create new Feature called IMF for all patients.

![image](https://github.com/user-attachments/assets/13842b63-a7d8-42df-8f13-88538d9238ec)



### Based on ORA
Over-representation analysis (ORA) is used to determine which a priori defined gene sets are more present (over-represented) in a subset of “interesting” genes than what would be expected by chance. To infer functional enrichment scores, we will run the ORA method. As input data, it accepts an expression matrix. The top 5% of expressed genes by sample are selected as the set of interest. The final score is obtained by log-transforming the obtained p-values, meaning that higher values are more significant.
We then map the score into pathways according to the annotated gene sets in Molecular Signatures Database (MSigDB). Users can choose different gene sets to compare among cohorts, for example Hallmark or Reactome. 

#### Step 1: In our example, we choose Reactome and cell type signature.

![image](https://github.com/user-attachments/assets/35bb445d-b7cd-4ad8-8a3d-953dbb193871)

Here you can have options, which label will be used for differentiate, and which label will be used for plot. Since we want to find the similarity for drug response between two different drugs. We choose LABEL for differentiate and LABEL1 for plot.

#### Step 2: The next picture shows after processing, it shows the differentiated pathways similar to DEGs between response and non-response groups.

![image](https://github.com/user-attachments/assets/84f8a7b8-346f-48f0-9e2f-a3c0de21e6fa)

#### Step 3: Then you can select a suitable threshold for the target plot.
From the plot, we can see the difference and similarity between differnt values of LABEL1.

![image](https://github.com/user-attachments/assets/86fe142e-53c4-4543-a0cb-97dff8857099)

### Association analysis
Here, the user has the option to associate selected clinical data parameters with the (harmonised) omics data in order to study phenotypes. We provide Random Forests, XGBoost and LightGBM with its corresponding parameters for basic tuning. The data will be split into training and test (20% default) datasets. AUC-ROC or MSE will be shown according to different cases. This module will provide SHAP feature importance plot and SHAP dependence plot.

![image](https://github.com/user-attachments/assets/a274c4d5-c98f-42c3-9c50-fc012214fa05)

![image](https://github.com/user-attachments/assets/37ff9341-b720-4131-85f4-26f52cab72b8)

![image](https://github.com/user-attachments/assets/7a0e40fb-7f22-4df1-b95a-97922062e55f)

Then you can check metagene_6_'s enrichment analysis result in Enrichment Analysis tab.

![image](https://github.com/user-attachments/assets/bcf6bcdd-fac3-44b3-ad9a-b7e0a329c939)

### Causal Analysis
In the option, users can do causal analysis on treatment efficacy by reducing the selection bias between treatment and control group and handle heterogeneity among patients.
The platform provides Causal Forests based on PCA/ICA processed components for cohorts. Causal Forests is a machine learning method used for estimating heterogeneous treatment effects (HTE) in observational data. It is an extension of Random Forests designed specifically for causal inference. 
The reason why we provide both PCA and ICA processed components is because we think when focusing on HTE, we think PCA will be more robust. However, when focusing on explainable components, ICA will be better.
When using this module, user needs to designate T and Y as the parameters. T is the label representing treatment effect or the predefined cohorts, which we limit only for categorical variables; Y is the dependent variable for phenotype.

#### Step 1: Fist stage, settings for T&Y models.
![image](https://github.com/user-attachments/assets/2990084e-950f-47e4-9023-6a1fec01d654)

#### Step 2: Result for T&Y models.
![image](https://github.com/user-attachments/assets/409494bb-7113-4ff2-a4cf-300bf8d79359)

#### Step 3: Second stage, get HTE for test dataset
After runing the first stage to get T&Y models, then run the second stage. You will see the estimated HTE on test dataset.

![image](https://github.com/user-attachments/assets/9ba46945-6c69-40bc-a8f2-481d4336915c)

#### Step 4: Show SHAP value of Feature Importance.
Then we can see the Feature Importance plot.

![image](https://github.com/user-attachments/assets/b1b1452d-2115-4d5f-ba9d-c4774ae4f043)

As we see, old aged people is good for anti-TNF treatment. High M is good for anti-TNF treatment. We can also check the position of metagene_6_ which is IFN gamma.

![image](https://github.com/user-attachments/assets/d2ba45fa-9546-4883-a46b-d1931651e42e)

Metagene_1_ is L1CAM which is related to **pain**, but **Note, we just use RF, and the AUC for Y is not good enough.**

### DML
After that we generialised the algorithm to double machine learning (DML), which means two machine learning models to overcome Causal Forests’ limitation.
We then extend Causal Forests and DML to the similarity exploration across different tissues, omics levels and diseases. Under this situation, data harmalisation step is not necessary, which will avoid from potentially removing the biological meaning.

**Note, before running DML, user need to do cohort-ICA first, and it is compulsory.**
#### Step 1: Do cohort-ICA.

![image](https://github.com/user-attachments/assets/c14811b5-f3e3-4e91-bda9-6b9904536e00)

### Step 2: DML module & Param settings.
![image](https://github.com/user-attachments/assets/83266ef4-ba82-4dee-a6c6-814def68e731)

#### 2-1. SHAP plot for model I.
![image](https://github.com/user-attachments/assets/b5dbf046-8683-4f10-a988-da35d621a156)

#### 2-2. SHAP plot for model II.
![image](https://github.com/user-attachments/assets/29db091b-ab23-486b-9db3-534aa229b8d9)

#### 2-3. Combined SHAP plot/Report.
![image](https://github.com/user-attachments/assets/ccafa93b-1d33-4e7c-9ddc-b2a3e8b3bd5e)

![image](https://github.com/user-attachments/assets/791427c2-67d7-4783-bf7a-c0b598cf4305)

Here, we can see c_agvas is the most important feature for 2 models and metagene_5_Ant is the most important component.
However, unfortunately, when doing enrichment analysis, we didn't find the pathways for metagene_5_.
One possible explaination is maybe it is still unknown, or ICA make a wrong decomposition. But compared to c_agvas, its importance is very low.
**Here we have another trick: by making c_agvas as a label to exclude c_agvas. Let's see what we can do.**

#### 2-4. Making c_agvas as a label and exclude it.
Under 'Data Processing' tab, we make c_agvas as a label, similar to RNAcare, separating patients with c_agvas into 2 subgroups: low and high. Then we need to activate the label!

![image](https://github.com/user-attachments/assets/37dfde53-9bb6-4157-9378-9275e1f89265)

**Activate the created label**

![image](https://github.com/user-attachments/assets/542295c3-c1a4-4318-ad06-2d2d4a719578)

After that we run DML again and let's see the result.

For Model I:

![image](https://github.com/user-attachments/assets/b57e34a8-3de5-43b6-a25c-89fb18228fc8)

For Model II:

![image](https://github.com/user-attachments/assets/ca481991-bfd7-4805-9b0b-aa279b860e10)

For combined report:

![image](https://github.com/user-attachments/assets/4ad32902-231c-4767-974b-ee88071be2ef)

Now, we can see after excluding c_agvas,  metagene_5_Ant became more important. Same measure can be done until user finds the meanful metagene.

So after we exclude c_das as well, we find what we are looking for.

![image](https://github.com/user-attachments/assets/fcff466c-d65d-46ce-b5ff-428dc55b1dcf)

We do Enrichment analysis for metagene_2_Ant and find it is enriched in REACTOME_NONSENSE_MEDIATED_DECAY_NMD.

![image](https://github.com/user-attachments/assets/4a3d4020-0b0b-4a27-a9dd-83db3ed098cd)











































