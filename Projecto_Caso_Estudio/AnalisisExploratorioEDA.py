import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap as um
import math
import statistics
import seaborn as sns
pd.options.display.max_rows = 10
import warnings
warnings.filterwarnings('ignore')


class AnalisisDatosExploratorio():
    def __init__(self, path, num):
        self.__df = self.__cargarDatos(path, num)  
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
            
    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include = ["number"])

    def analisisCompleto(self):  
        self.__df = pd.get_dummies(self.__df)
            
    def __cargarDatos(self, path, num):
        if num == 1:
            return pd.read_csv(path,
            sep = ",",
            decimal = ".",
            index_col = 0)
        if num == 2:
            return pd.read_csv(path,
            sep = ";",
            decimal = ".")
     
    def analisis(self):
      print("Dimensiones:", self.__df.shape)
      print(self.__df.head())
      print(self.__df.describe())

      print("Media:")
      print(self.__df.mean(numeric_only=True))

      print("Mediana:")
      print(self.__df.median(numeric_only=True))

      print("Desviación estándar:")
      print(self.__df.std(numeric_only=True, ddof=0))

      print("Máximo:")
      print(self.__df.max(numeric_only=True))

      print("Mínimo:")
      print(self.__df.min(numeric_only=True))

      print("Cuantiles:")
      print(self.__df.quantile(np.array([0,.33,.50,.75,1]), numeric_only=True))

      self.__graficosBoxplot()
      self.__funcionDensidad()
      self.__histograma()
      self.__correlaciones()
      self.__graficoDeCorrelacion() 
        
    def __graficosBoxplot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        boxplots = self.__df.boxplot(return_type='axes',ax=ax)
        plt.show()
          
    def __funcionDensidad(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,8), dpi = 200)
        densidad = self.__df[self.__df.columns].plot(kind='density',ax = ax)
        plt.show()
                 
    def __histograma(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,6), dpi = 200)
        densidad = self.__df[self.__df.columns].plot(kind='hist', ax = ax)
        plt.show()
      
    def __correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print(corr)

    def __graficoDeCorrelacion(self):
        fig, ax = plt.subplots(figsize=(12, 8), dpi = 150)
        paleta = sns.diverging_palette(220, 10,as_cmap=True).reversed()
        corr = self.__df.corr(numeric_only=True)
        sns.heatmap(corr, vmin= -1, vmax=1, cmap= paleta,square=True, annot=True, ax=ax)
        plt.show()

    
