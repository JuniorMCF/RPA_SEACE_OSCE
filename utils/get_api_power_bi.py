# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:28:16 2022
@author: Castillo Flores Junior Manuel

Embed this python script in Power BI 
to extract the data source from the Laravel API, 
change the environment variables to the variables 
used for your laravel api API_URL , EMAIL_API and PASS_API

"""
import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

url_domain = os.getenv('API_URL')

data     = {'email':os.getenv('EMAIL_API'),'password': os.getenv('PASS_API')}
response  = requests.post( url_domain  + "login", data = data)

result = response.json()["data"]

token = result["token"]
        
headers = {'Authorization' : 'Bearer '+token,'Accept':'application/json','Content-Type':'application/json'}


r1 = requests.get( url_domain + "osce",headers = headers )
r2 = requests.get( url_domain + "entidad",headers = headers )
r3 = requests.get( url_domain + "entidad-contratante",headers = headers )
r4 = requests.get( url_domain + "cronograma",headers = headers )
r5 = requests.get( url_domain + "contrato",headers = headers )


osce = pd.DataFrame(r1.json())
entidades= pd.DataFrame(r2.json())
contratantes= pd.DataFrame(r3.json())
cronogramas= pd.DataFrame(r4.json())
contratos= pd.DataFrame(r5.json())

print(osce)
print(entidades)
print(contratantes)
print(cronogramas)
print(contratos)