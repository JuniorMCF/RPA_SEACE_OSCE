# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:51:45 2022

@author: Castillo Flores Junior Manuel
@script: RPA for OSCE-SEACE contrataciones con el estado
"""

from tk_departments import *
from selenium_web_driver import SeleniumWebDriver
import os
from dotenv import load_dotenv

load_dotenv()


departments = [
    "AMAZONAS",
    "ANCASH",
    "APURIMAC",
    "AREQUIPA",
    "AYACUCHO",
    "CAJAMARCA",
    "CALLAO",
    "CUSCO",
    "EXTERIOR",
    "HUANCAVELICA",
    "HUANUCO",
    "ICA",
    "JUNIN"
    "LA LIBERTAD",
    "LAMBAYEQUE",
    "LIMA",
    "LORETO",
    "MADRE DE DIOS",
    "MOQUEGUA",
    "MULTIDEPARTAMENTAL",
    "PASCO",
    "PIURA",
    "PUNO",
    "SAN MARTIN",
    "TACNA",
    "TUMBES",
    "UCAYALI"
    
    ]


if __name__ == "__main__":
    
    driver = SeleniumWebDriver( os.getenv('OSCE_URL')  )                 #observer

    tk_interface = TkDepartment( departments , driver )                  #observable

    