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

#### Step 9(optional): Nginx setting
Edit your Nginx configuration file (e.g., /etc/nginx/sites-available/your_site.conf) and ensure it includes the following directive inside the appropriate location or server block:
```bash
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Real-IP $remote_addr;
```
Reload Nginx to apply the changes:`sudo systemctl reload nginx`
