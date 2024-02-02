#NOMBRES:
#DANIELA HERRERA
#PAULA MAYO
#WILSON MORALES
############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from scipy.stats import shapiro
from scipy.stats import boxcox
import itertools
#Importa la clase ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch

####################################################
real = pd.read_excel("API_EN.ATM.CO2E.KT_DS2_es_excel_v2_6301161.xls", header=3)  # Ignorar las primeras 3 filas
# Mostrar las primeras filas para verificar la carga correcta
print(real.head())

# Lista de países de Latinoamérica
paises_latinoamerica = ['Argentina', 'Bolivia', 'Brasil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras', 'México', 'Nicaragua', 'Panamá', 'Paraguay', 'Perú', 'República Dominicana', 'Uruguay', 'Venezuela']

# Filtrar el DataFrame para incluir solo los países de Latinoamérica
df_latinoamerica = real[real['Country Name'].isin(paises_latinoamerica)]

# Mostrar las primeras filas del DataFrame filtrado
print(df_latinoamerica.head())
#########################################################################################
#########################################################################################
# Filtrar el DataFrame para incluir solo el año 2020
df_2020 = df_latinoamerica[['Country Name', '2020']]

# Calcular el promedio de los valores para todos los países en el año 2020
valor_promedio_2020 = df_2020['2020'].mean()

# Mostrar el valor promedio
print("El valor promedio del indicador seleccionado entre los países de América Latina en el año 2020 es:", valor_promedio_2020)

# Seleccionar solo las columnas que contienen los datos de las fechas
fechas = df_latinoamerica.columns[4:]
print(fechas)
# Iterar sobre los países de Latinoamérica y trazar un gráfico para cada uno
for index, row in df_latinoamerica.iterrows():
    pais = row['Country Name']
    datos_pais = row[fechas]
    plt.plot(fechas, datos_pais, label=pais)

# Agregar etiquetas y leyenda al gráfico
plt.xlabel('Año')
plt.ylabel('CO2 (kt)')
plt.title('Emisión de CO2 por país en Latinoamérica (1960-2020)')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=90)
# Mostrar el gráfico
plt.tight_layout()
plt.show()
####################################################################
####################################################################
# Filtrar el DataFrame para incluir solo los últimos 5 años de datos
ultimos_5_anios = df_latinoamerica.iloc[:, -5:]

# Calcular la matriz de correlación
correlacion = ultimos_5_anios.corr()

# Crear el mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Mapa de Correlación entre los últimos 5 años de datos para países de América Latina')
plt.xlabel('Año')
plt.ylabel('Año')
plt.show()
######################################################################################
######################################################################################

