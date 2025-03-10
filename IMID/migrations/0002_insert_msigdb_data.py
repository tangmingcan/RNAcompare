from django.db import migrations
import pandas as pd

def insert_msigdb_data(apps, schema_editor):
    Msigdb = apps.get_model('IMID', 'Msigdb')  # Replace 'IMID' with your actual app name
    # Load the CSV file using pandas
    df = pd.read_csv('msigdb.csv')
    
    # Loop through the DataFrame and insert the data
    for _, row in df.iterrows():
        Msigdb.objects.create(
            symbol=row['genesymbol'],
            collection=row['collection'],
            geneset=row['geneset']
        )
        
def delete_msigdb_data(apps, schema_editor):
    Msigdb = apps.get_model('imid', 'Msigdb')
    Msigdb.objects.all().delete()
    
class Migration(migrations.Migration):

    dependencies = [
        ("IMID", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(code=insert_msigdb_data, reverse_code=delete_msigdb_data),
    ]
