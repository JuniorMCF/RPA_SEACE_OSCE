# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:36:05 2022

@author: Castillo Flores Junior Manuel
"""
from selenium_web_driver import WebDriver
import os
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    
    webDriver = WebDriver( os.getenv('OSCE_URL') )
    webDriver.openChrome()
    
    # bucle for extract captcha images for training,
    # after downloading the images tag them for training
    for i in range(1,200):
        title = 'captcha'+str(i)
        webDriver.getImagesCaptchaForTraining(title)