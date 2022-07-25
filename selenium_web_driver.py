# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:54:21 2022

@author: Castillo Flores Junior Manuel

        id_tabla info general entidad -> tbFicha:j_idt895
        id_tabla info general procedimiento -> tbFicha:j_idt919
        id_tabla cronomgrama id tbody -> tbFicha:dtCronograma_data
        id_tabla entidad contratante id tbody -> tbFicha:dtEntidadContrata_data
        id_tabla participantes contratados id tbody -> tbFicha:idGridLstItems:0:dtParticipantes_data
    
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
import time
from api import LaravelSanctumApi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pattern_observer import *
from tk_departments import *

import numpy as np


class SeleniumWebDriver(Observer):

    def __init__(self, url ,name="Observer"):
        Observer.__init__(self, name)
        
        self._url = url
        self._path_chrome_driver        = './driver/chromedriver.exe'
        
        #variables for collections data
        self._data_osce                 = []
        self._general_information       = []
        self._process                   = []
        self._cronogramas               = []
        self._entidades_contratantes    = []
        self._contratos                 = []
        self._row_number                = 0                 # to count the rows that we go through every 15 it restarts
        self._number_page_detail        = 0
        self._not_finish                = True
        self._status                    = ""
        
        #define ids and xpath
        self._catpcha_image                     =   "tbBuscador:idFormBuscarProceso:captchaImg"
        self._element_captcha_input             =   "tbBuscador:idFormBuscarProceso:codigoCaptcha"
        self._element_button_filter             =   "tbBuscador:idFormBuscarProceso:btnBuscarSel"
        self._element_table_body_contracts      =   "tbBuscador:idFormBuscarProceso:dtProcesos_data"
        self._element_next_page                 =   "ui-paginator-next" 
        self._element_table_entity              =   "tbFicha:j_idt895"                          #varia por el framework backend jsp
        self._element_table_process             =   "tbFicha:j_idt919"                          #varia por el framework backend jsp
        self._element_legend                    =   '//*[@id="tbFicha:j_idt1204"]/legend'       #varia por el framework backend jsp
        self._element_status_contract           =   '//*[@id="tbFicha:idGridLstItems_content"]/table/tbody/tr[1]/td/table[2]/tbody/tr[2]/td[8]'
        self._element_timeline                  =   "tbFicha:dtCronograma_data"
        self._element_contract_entitys          =   "tbFicha:dtEntidadContrata_data"
        self._element_contracts                 =   "tbFicha:idGridLstItems:0:dtParticipantes_data"
        
        
        self.open_chrome()
        
        
        
    def notify_changes(self, change):
        # here receive data from observable change.data is department
        print(f"{self._name} receive changes from {change._name} : {change._data}")
        
        result_solve_captcha = self.solve_captcha(change._data)
        
        if result_solve_captcha == False:
            self._driver.close()
        else: 
            self.collect_data(change._data)
        
    def open_chrome(self):
        try:
            self._driver = webdriver.Chrome(
                self.get_resource_path(self._path_chrome_driver))
            self._driver.get(self._url)
        except Exception as e:
            print("Error in web driver instance ", e)

    def get_resource_path(self, path_driver):
        try:
            base_path = sys._MEIPASS
        except Exception as e:
            base_path = os.path.dirname(__file__)
        return os.path.join(base_path, path_driver)

    def solve_captcha(self, department):
        self.get_catpcha_image_from_web()  # capture and save image captcha in directory
        # load ocr model for caracter recognition
        model = keras.models.load_model('ocr_model.h5')
        # read captcha image for tensorflow
        img = tf.io.read_file("osce_captcha.PNG")
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [50, 200])
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.expand_dims(img, axis=0)
        preds = model.predict(img)
        pred_text = self.decode_batch_predictions(preds)
        print("prediccion "+str(pred_text[0]))

        is_correct_captcha = self.get_table(pred_text[0],department)
        
        return is_correct_captcha #Boolean

    def get_table(self,captcha,department):
        self._driver.execute_script("document.getElementById('tbBuscador:idFormBuscarProceso:departamento_label').innerHTML = '"+department+"';")
        self._driver.execute_script("Array.from(document.getElementById('tbBuscador:idFormBuscarProceso:departamento_input').options).forEach(function(option_element) { if(option_element.text == '"+department+"'){  option_element.setAttribute('selected' , 'selected');   }else{ option_element.removeAttribute('selected') }});")
        input_captcha = self._driver.find_element(By.ID, self._element_captcha_input)
        input_captcha.clear()
        input_captcha.send_keys(captcha)
        time.sleep(1)
        buton_form = WebDriverWait(self._driver, 2).until(EC.presence_of_element_located((By.ID, self._element_button_filter)))
        # or just wait for a second for browser(driver) to change
        buton_form.click()
        time.sleep(3)
        tbody = self._driver.find_element(By.ID, self._element_table_body_contracts)
        
        if(tbody.text == "No se encontraron Datos"):
            return False
        else:
            self._driver.refresh()                          #refreshing page for extracting data
            return True     

    def collect_data(self, department):
      
        while(self._not_finish):
            try:
                if self._row_number % 15 == 0 and self._row_number >= 15:
                    
                    self.collect_data_table_osce(department)
                    
                    self._row_number = 0            # reset for another pickup

                    self.next_page()

                    continue

                self.get_row_data_and_navigate_to_next_view()
    
                self.get_table_entity()
                
                self.get_table_process()
                
                self.get_table_timeline()
                
                self.get_table_contract_entitys()

                self.get_table_contracts()
                
                self.get_status_contract()

                self._driver.back()
                
                time.sleep(3)
                
            except NoSuchElementException as error:
                print( f"{error.message}" ) 
                self._not_finish = False
    
    def collect_data_table_osce(self,department):
        tbody = self._driver.find_element(By.ID, self._element_table_body_contracts)
        parsed_html = BeautifulSoup(tbody.get_attribute('innerHTML'), "html.parser")
        self._data_osce = []
        for td in parsed_html.find_all('tr'):
            row = [i.text for i in td.find_all('td')]
            row.append(department)
            self._data_osce.append(row)
        
        self.sendToBackend( self._data_osce, self._general_information, self._timeline,  self._contract_entitys, self._contracts)
    
    def next_page(self):
        next_page_link = WebDriverWait(self._driver, 3).until(EC.presence_of_element_located((By.CLASS_NAME, self._element_next_page )))
        if "ui-state-disabled" in next_page_link.get_attribute("class"): #last page
            self._not_finish = False
        next_page_link.click()
        time.sleep(4)
        
    def get_row_data_and_navigate_to_next_view(self):
        # buscar la nomenclatura        
        col_3 = self._driver.find_element(By.XPATH, '//*[@id="tbBuscador:idFormBuscarProceso:dtProcesos_data"]/tr['+str(self._row_number+1)+']/td[4]').text
        col_4 = self._driver.find_element(By.XPATH, '//*[@id="tbBuscador:idFormBuscarProceso:dtProcesos_data"]/tr['+str(self._row_number+1)+']/td[6]').text
        self._nomenclatura = col_3 + "|" + col_4  # nomenclatura y objeto contratacion
        detalle_button = self._driver.find_element(By.ID, "tbBuscador:idFormBuscarProceso:dtProcesos:"+str(self._number_page_detail)+":grafichaSel")
        self._row_number += 1
        self._number_page_detail += 1
        WebDriverWait(detalle_button.click(), 5)
        time.sleep(6)
        
    def get_table_entity(self):
        try:
            t_entity = self._driver.find_element( By.ID , self._element_table_entity)          
            tbody_entity = t_entity.find_element( By.TAG_NAME, "tbody")
            self._general_information = self.transform_entity_to_data_frame(tbody_entity, self._general_information, self._nomenclatura, self._status)
        except NoSuchElementException as error:
            print( f"{error.message}" ) 
            pass
            
    def get_table_process(self):
        try:
            table_process = self._driver.find_element(By.ID,self._element_table_process )           
            tbody_process = table_process.find_element(By.TAG_NAME, "tbody")
            self._process = self.transform_entity_to_data_frame_without_bases(tbody_process, self._process, self._nomenclatura, self._status)
        except NoSuchElementException as error:
            print( f"{error.message}" ) 
            pass

    def get_table_timeline(self):
        try:
            self._table_timeline = self._driver.find_element(By.ID, self._element_timeline)
            self._timeline = self.transform_data_frame(self._table_timeline, self._timeline, self._nomenclatura, self._status)
        except NoSuchElementException as error:
            print( f"{error.message}" ) 
            pass
    def get_table_contract_entitys(self):
        try:
            table_contract_entitys = self._driver.find_element(By.ID, self._element_contract_entitys)
            self._contract_entitys = self.transform_data_frame(table_contract_entitys, self._contract_entitys, self._nomenclatura, self._status)
        except NoSuchElementException as error:
            print( f"{error.message}" ) 
            pass
    
    def get_table_contracts(self):
        try:
            table_contracts = self._driver.find_element(By.ID,self._element_contracts)
            self._contracts = self.transform_data_frame(table_contracts, self._contracts, self._nomenclatura, self._status)
        except NoSuchElementException as error:
            print( f"{error.message}" ) 
            pass
    
    def get_status_contract(self):
        element_legend = self._driver.find_element(By.XPATH, self._element_legend)  
        element_legend.click()
        time.sleep(1)
        
        element_status = self._driver.find_element(By.XPATH, self._element_status_contract)
        self._status = element_status.text
        print(self._status)

    def get_catpcha_image_from_web(self):
        try:
            with open('osce_captcha.png', 'wb') as file:
                # identify image to be captured
                l = self._driver.find_element(By.ID, self._captcha_image)
                file.write(l.screenshot_as_png)
            img = open('osce_captcha.png')
            return img
        except Exception as e:
            print("Error in get captcha image ", e)

    def transform_data_frame(self, tbody, data, nomenclatura, estado):

        parsed_html = BeautifulSoup(tbody.get_attribute('innerHTML'), "html.parser")

        for td in parsed_html.find_all('tr'):
            row = []
            isEmtpy = False
            for i in td.find_all('td'):
                if i.text != 'No se encontraron Datos':
                    row.append(i.text)
                    isEmtpy = False
                else:
                    isEmtpy = True
            if(isEmtpy == False):
                row.append(nomenclatura)
                row.append(estado)
                
            if len(row) > 0:
                data.append(row)
      
        return data

    def transform_entity_to_data_frame(self, tbody, data, nomenclatura, estado):
        parsed_html = BeautifulSoup(
            tbody.get_attribute('innerHTML'), "html.parser")
        row = []
        for td in parsed_html.find_all('tr'):
            count = 0
            for i in td.find_all('td'):
                if i.text != 'Lugar y cuenta de pago del costo de Reproducción de las Bases':
                    if count == 1 and i.text != '\n\n\nBanco\nCuenta\n\n\nCaja de la Entidad\n\n\n' and i.text != 'Cuenta':
                        row.append(i.text)
                count += 1
        row.append(nomenclatura)
        row.append(estado)
        data.append(row)
        return data

    def transform_entity_to_data_frame_without_bases(self, tbody, data, nomenclatura, estado):

        parsed_html = BeautifulSoup(tbody.get_attribute('innerHTML'), "html.parser")
        row = []
        for td in parsed_html.find_all('tr', {"class": "ui-widget-content"}):
            count = 0
            prev = ''
            for i in td.find_all('td'):
                if count % 2 != 0:
                    if prev != 'Monto del costo de Reproducción de las Bases:' and prev != 'Lugar y cuenta de pago del costo de Reproducción de las Bases:':
                        row.append(i.text)
                else:
                    prev = i.text
                count += 1
        row.append(nomenclatura)
        row.append(estado)
        data.append(row)
        return data

    def sendToBackend(self, data_osce, general_information, timeline, contract_entitys, contracts):
        # osce data
        df_osce_actual_year = pd.DataFrame(columns=['Numero',
                                                    'Nombre o Sigla de la Entidad',
                                                    'Fecha y Hora de Publicacion',
                                                    'Nomenclatura',
                                                    'Reiniciado Desde',
                                                    'Objeto de Contratación',
                                                    'Descripción de Objeto',
                                                    'Código SNIP',
                                                    'Código Unico de Inversion',
                                                    'Valor Referencial / Valor Estimado',
                                                    'Moneda',
                                                    'Versión SEACE',
                                                    'Acciones',
                                                    'Departamento'
                                                    ], data= data_osce)

        print(df_osce_actual_year)

        # entidad contratante, direccion legal, pagina web, telefono de entidad
        df_entitys = pd.DataFrame(columns=['Entidad contratante',
                                             'Dirección legal',
                                             'Página web',
                                             'Teléfono de la entidad',
                                             'Nomenclatura',
                                             'estado',
                                             ], data=general_information)

        # objeto contratacion, descripcion, valor estimado referencial, monto del derecho de participacion, monto costo de reproduccion de las bases ,cuenta de pago y lugar , fecha publicacion
        # df_procedimientos = pd.DataFrame(columns=['Objeto contratación',
        #                                            'Descripcion',
        #                                            'Valor estimado referencial',
        #                                            'Monto del derecho de participación',
        #                                            #'Monto costo de reproduccion de las bases',
        #                                            #'Cuenta de pago y lugar ',
        #                                            'Fecha publicación',
        #                                            'Nomenclatura'],data = convocatorias_info_general_procedimiento)

        print(df_entitys)
        # etapa , fecha inicio , fecha fin
        df_timeline = pd.DataFrame(columns=['Etapa',
                                            'Fecha inicio',
                                            'Fecha fin',
                                            'Nomenclatura',
                                            'estado',
                                            ], data=timeline)
        # df_cronogramas.to_excel("cronogramas_osce_v3.xlsx")

        # ruc ,entidad contratante
        df_contract_entitys = pd.DataFrame(columns=['RUC',
                                                    'Entidad contratante',
                                                    'Nomenclatura',
                                                    'estado', ], data=contract_entitys)
        # df_entidades_contratantes.to_excel("entidades_cotnratantes_osce_v3.xlsx")
        print(df_contract_entitys)

        # postor, mype , ley de  promocion de la selva, bonificacion colindante,cantidad adjudicada, monto adjudicado
        df_contracts = pd.DataFrame(columns=['postor',
                                             'mype',
                                             'ley de  promocion de la selva',
                                             'bonificacion colindante',
                                             'cantidad adjudicada',
                                             'monto adjudicado',
                                             'Nomenclatura',
                                             'estado', ], data=contracts)
        # df_contratos.to_excel("contratos_osce_v3.xlsx")
        print(df_contracts)

        print("Enviando datos")

        api = LaravelSanctumApi()
        api.sendOsceTable(df_osce_actual_year)
        api.sendOsceEntitys(df_entitys)
        api.sendOsceContractEntitys(df_contract_entitys)
        api.sendOsceTimeline(df_timeline)
        api.sendOsceContracts(df_contracts)

        print("-------Finalización de carga de datos--------")

    def decode_batch_predictions(self, pred):
        with open("vocab.txt", "r") as f:
            vocab = f.read().splitlines()

        # Mapping integers back to original characters
        num_to_char = layers.StringLookup(
            vocabulary=vocab, mask_token=None, invert=True
        )

        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :5
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(
                num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
    
    def get_captcha_for_training(self, title):
        try:
            with open('./training/'+title+'.png', 'wb') as file:
                # identify image to be captured
                l = self._driver.find_element(By.ID, self._catpcha_image )
                file.write(l.screenshot_as_png)
            self._driver.refresh()
        except Exception as e:
            print("Error in get captcha image ", e)
