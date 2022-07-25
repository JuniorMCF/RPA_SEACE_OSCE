# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:40:33 2022

@author: Castillo Flores Junior Manuel
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
from pattern_observer import *
from threading import Thread

class TkDepartment(Observable):

    def __init__(self, departments , driver , name = "observable"):
        # 
        # Observable by selenium object for tk_department
        #
        Observable.__init__(self, name)
        self._driver = driver
        self.attach_observer(self._driver)
        
        
        try:
            self._root = Tk()
            self._root.title("RPA")
            self._root.resizable(width=False, height=False)

            self._label_department = Label(self._root,  text="Departamento:")
            self._label_department.grid(pady=5, row=1, column=0)

            self._combo = Combobox(self._root, width=40)
            self._combo['values'] = departments
            self._combo.current(1)  # set the selected item
            self._combo.grid( padx=5 , row=1 , column=1 )

            self._button_accept = Button(self._root, text="Iniciar recolección", width=66, command= lambda: self.solve_captcha())
            self._button_accept.grid( padx=10, pady=10, row=2, column=0, columnspan=2)
            self._root.mainloop()
        except Exception as e:
            print(e)
            pass

    def solve_captcha(self):
        
        department = self._combo.get()
        self._root.destroy()
        thread = Thread(target =  self.notify_changes(department))
        thread.start()
        
        
       