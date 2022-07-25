# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:11:04 2022

@author: Castillo Flores Junior Manuel
"""
import requests
import json
import time
import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class LaravelSanctumApi:
    url_domain = os.getenv('API_URL')
    headers = {}

    def __init__(self):
        self.authenticateUser()

    def authenticateUser(self):
        response = requests.post(self.url_domain+'login', data={
            'email': os.getenv('EMAIL_API'),
            'password': os.getenv('PASS_API')})
        data = response.json()["data"]
        token = data["token"]

        self.headers = {'Authorization': 'Bearer '+token,
                        'Accept': 'application/json', 'Content-Type': 'application/json'}

    def sendOsceTable(self, df_osce):
        for ind in df_osce.index:
            unique = df_osce['Nomenclatura'][ind] + "|" + \
                df_osce['Objeto de Contratación'][ind]
            fecha_publicacion = df_osce['Fecha y Hora de Publicacion'][ind].split(
                " ")
            fecha_pub = '' if fecha_publicacion[0].strip() == '' else datetime.datetime.strptime(
                fecha_publicacion[0].strip(), '%d/%m/%Y').strftime('%Y-%m-%d %H:%M:%S')
            data = json.dumps({
                'nombre_sigla_entidad': df_osce['Nombre o Sigla de la Entidad'][ind],
                'fecha_publicacion': fecha_pub,
                'nomenclatura': unique,
                'reiniciado_desde': df_osce['Reiniciado Desde'][ind],
                'objeto_contratacion': df_osce['Objeto de Contratación'][ind],
                'descripcion': df_osce['Descripción de Objeto'][ind],
                'codigo_snip': df_osce['Código SNIP'][ind],
                'codigo_unico_inversion': df_osce['Código Unico de Inversion'][ind],
                'valor_estimado': self.correctNumberFormat(df_osce['Valor Referencial / Valor Estimado'][ind]),
                'moneda': df_osce['Moneda'][ind],
                'version_seace': df_osce['Versión SEACE'][ind],
                'acciones': df_osce['Acciones'][ind],
                'departamento': df_osce['Departamento'][ind]
            })

            time.sleep(1)
            response = requests.post(
                self.url_domain+'save/osce', data=data, headers=self.headers)
            print(response.status_code)

    def sendOsceEntitys(self, df_entidades):
        for ind in df_entidades.index:

            time.sleep(2)
            data = json.dumps({
                'entidad_contratante': df_entidades['Entidad contratante'][ind],
                'direccion_legal': df_entidades['Dirección legal'][ind],
                'pagina_web': df_entidades['Página web'][ind],
                'telefono': df_entidades['Teléfono de la entidad'][ind],
                'nomenclatura': df_entidades['Nomenclatura'][ind],
                "estado": df_entidades['estado'][ind],
            })
            response = requests.post(
                self.url_domain+'save/entidad', data=data, headers=self.headers)
            print(response.status_code)

    def sendOsceContractEntitys(self, df_e_contratantes):
        for ind in df_e_contratantes.index:
            time.sleep(2)
            ruc = str(df_e_contratantes['RUC'][ind])

            data = json.dumps({
                'RUC': ruc,
                'entidad_contratante': df_e_contratantes['Entidad contratante'][ind],
                'nomenclatura': df_e_contratantes['Nomenclatura'][ind],
                'estado': df_e_contratantes['estado'][ind],
            })

            response = requests.post(
                self.url_domain+'save/entidad-contratante', data=data, headers=self.headers)
            print(response.status_code)

    def sendOsceTimeline(self, df_cronogramas):
        for ind in df_cronogramas.index:
            time.sleep(2)

            fecha_inicio = df_cronogramas['Fecha inicio'][ind].split(" ")
            f_ini = datetime.datetime.strptime(
                fecha_inicio[0].strip(), '%d/%m/%Y').strftime('%Y-%m-%d %H:%M:%S')
            fecha_fin = df_cronogramas['Fecha fin'][ind].split(" ")
            f_fin = datetime.datetime.strptime(
                fecha_fin[0].strip(), '%d/%m/%Y').strftime('%Y-%m-%d %H:%M:%S')
            response = requests.post(self.url_domain+'save/cronograma', data=json.dumps({
                'etapa': df_cronogramas['Etapa'][ind],
                'fecha_inicio': f_ini,
                'fecha_fin': f_fin,
                'nomenclatura': df_cronogramas['Nomenclatura'][ind],
                'estado': df_cronogramas['estado'][ind],
            }), headers=self.headers)

            print(response.status_code)

    def sendOsceContracts(self, df_contratos):
        for ind in df_contratos.index:
            time.sleep(2)
            response = requests.post(self.url_domain+'save/contrato', data=json.dumps({
                'postor': df_contratos['postor'][ind],
                'mype': df_contratos['mype'][ind],
                'ley_promocion_de_selva': df_contratos['ley de  promocion de la selva'][ind],
                'bonificacion_colindante': df_contratos['bonificacion colindante'][ind],
                'cantidad_adjudicada': self.correctNumberFormat(df_contratos['cantidad adjudicada'][ind]),
                'monto_adjudicado': self.correctNumberFormat(df_contratos['monto adjudicado'][ind]),
                'nomenclatura': df_contratos['Nomenclatura'][ind],
                'estado': df_contratos['estado'][ind],
            }), headers=self.headers)

            print(response.status_code)

    # function for transform thousands format

    def correctNumberFormat(self, valor):
        if valor == "---":
            print("transform: "+str(valor))
            return 0.0
        else:
            print("transform: "+str(valor))
            if ',' not in valor:
                return valor
            else:
                return valor.replace(",", "")
