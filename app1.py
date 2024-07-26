# Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set
import warnings
warnings.filterwarnings('ignore')
import ast
import datetime as dt
from fastapi import FastAPI, HTTPException

# Función para cargar los datos:

app = FastAPI()

# Cargar los datos desde un archivo Parquet

def cargar_datos():
    # Ajusta la ruta al archivo Parquet
    df = pd.read_parquet('./Datasets/data_movies.parquet')
    df_credits = pd.read_parquet('./Datasets/data_credits.parquet')
    return df, df_credits

peliculas, creditos = cargar_datos()

@app.get("/cantidad_filmaciones_mes")


async def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    meses = {
        'enero': '01',
        'febrero': '02',
        'marzo': '03',
        'abril': '04',
        'mayo': '05',
        'junio': '06',
        'julio': '07',
        'agosto': '08',
        'septiembre': '09',
        'octubre': '10',
        'noviembre': '11',
        'diciembre': '12'
    }
    
    if mes not in meses:
        raise HTTPException(status_code=400, detail="Mes no válido")
    
    mes_num = meses[mes]
    
    try:
        peliculas['release_date'] = pd.to_datetime(peliculas['release_date'], errors='coerce') #Convierte la columna release_date a fecha (datetime)
        peliculas['release_month'] = peliculas['release_date'].dt.strftime('%m') #Extraigo el mes a la columna release_month
        conteo_peliculas_por_año = peliculas[peliculas['release_month'] == mes_num]['release_year'].value_counts().sort_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    resultado = conteo_peliculas_por_año.to_dict()
    return {"total_peliculas_por_año": resultado}

if __name__ == "__app1__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
