# En este proyecto realizamos algunos calculos estadisticos con Base de Datos Estructurados#
# Primero realizamos la importacion de las dependencias para que funcione#
import numpy as np
import pandas as pd
import math as m
import time as t
import matplotlib.pyplot as plt
import seaborn as sns
# Se ocupa por si no se quiere sacar los datos de forma local#
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Variables globales#
df = None
wait_time = 1.5 #Es en segundos

# Obtencion de Datos
## LOCAL
def load_data_local(ubicacion):
    try:
        df = pd.read_csv(ubicacion)
        return df
    except Exception as e:
        print(f"Error al cargar la B.D.: {e}")
        return None

## KAGGLE
# Para poder usarlo de forma correcta esta funcion ocupa conocer como regresa el 
# archivo kaggle y el nombre al que hace referencia la B.D. (se obtiene en la seccion
# de code)#
def load_data_cloud_kg(file_path,handle):
    try:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            handle,
            file_path
        )
        # print(df.head())
        return df

    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

# Actividad 1 #

#Funcion para obtencion de maximo de la columna #
def get_max_value(values):
    max_val = None
    # if(len(values) > 0):
    for p in values:
        if max_val is None or (isinstance(p,(int,float)) and p > max_val):
            max_val = p
    return max_val

#Funcion para obtencion de minimos de la columna#
def get_min_value(values): 
    min_value = None
    # if(len(values) > 0):
    for p in values:
        if min_value is None or (isinstance(p,(int,float)) and p < min_value):
            min_value = p
    return min_value

# Funcion para obtenccion de la media de una columna#
def get_mean(values):
    suma = 0
    contador = 0
    
    for p in values:
        if isinstance(p, (int, float)) and not m.isnan(p):
            suma += p
            contador += 1

    return suma / contador if contador > 0 else float("nan")


#Funcion para obtencion Desviacion estandar de la columna#
def get_desv_stand(values,mean):
    desv_stand = 0
    #Sumatoria de Resta de media y elemento al cuadado#
    #entre el total de elementos y al final raiz cuadrada#
    for v in values:
        if isinstance(v,(int,float)):
                desv_stand += m.pow(v-mean, 2)
    desv_stand = m.sqrt(desv_stand / len(values))

    return desv_stand

# Funcion para obtencion del numero de atributos (campos) #
def get_hM_atribute(df):
    columnas = df.shape[1]
    # Suponiendo que 'dataset' es tu matriz
    # num_filas = len(df)
    return columnas

# Funcion para obtencion del numero de instancias (filas) #
def get_hM_instance(values):
    # intances = 0
    # for v in values:
    #     intances = intances + 1
    # return intances
    return len(values)

# Observaciones de las columnas #
def get_observations(data_list):
    lista_nueva = []
    for valor in data_list:
        if valor not in lista_nueva:
            lista_nueva.append(valor)
    return lista_nueva

# Balance de clases (Porcentaje de los atributos de salida)#

def get_balanceofClasses(values):
    balance = {
        "Clase_A": 0,
        "Clase_B": 0,
        "Otros": 0
    }
    unicos = sorted(list(set(x for x in values if str(x).lower() != 'nan')))
    
    # 3. Validación: Si hay más de 2 clases, avisamos
    if len(unicos) > 2:
        print(f"Aviso: Se encontraron {len(unicos)} clases diferentes.")

    for v in values:
        # Si el valor es el primero del conjunto (ej. 0 o 1)
        if v == unicos[0]:
            balance["Clase_A"] += 1
        # Si el valor es el segundo del conjunto (ej. 1 o 2)
        elif len(unicos) > 1 and v == unicos[1]:
            balance["Clase_B"] += 1
        else:
            balance["Otros"] += 1
            
    # Renombramos las llaves para que el usuario sepa qué contó
    # Ejemplo: {"Positivo (2)": 10, "Negativo (1)": 5}
    resultado_final = {
        f"Valor ({unicos[0]})": balance["Clase_A"],
        f"Valor ({unicos[1] if len(unicos) > 1 else 'N/A'})": balance["Clase_B"]
    }
    
    if balance["Otros"] > 0:
        resultado_final["No validos"] = balance["Otros"]
        
    return resultado_final


# Actividad 2 #

def get_empty_Data(data_list):
    empty_count = sum(1 for x in data_list if pd.isna(x))
    porcentaje = (empty_count / len(data_list)) * 100
    return porcentaje

def calcular_estadisticas_outliers(data_list):
    if not data_list:
        return None

    # 1. Ordenar los datos (esencial para calcular cuartiles)
    data_ordenada = sorted(data_list)
    n = len(data_ordenada)

    # 2. Función interna para encontrar un percentil (interpolación lineal simple)
    def get_percentile(data, percentile):
        k = (len(data) - 1) * percentile
        f = int(k)
        c = k - f
        if f + 1 < len(data):
            return data[f] + (data[f+1] - data[f]) * c
        else:
            return data[f]

    # 3. Calcular Q1 (25%) y Q3 (75%)
    q1 = get_percentile(data_ordenada, 0.25)
    q3 = get_percentile(data_ordenada, 0.75)

    # 4. Calcular el Rango Intercuartílico (IQR)
    iqr = q3 - q1

    # 5. Calcular Límites Inferior y Superior
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    # 6. Identificar Outliers
    outliers = [x for x in data_list if x < limite_inferior or x > limite_superior]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inf": limite_inferior,
        "limite_sup": limite_superior,
        "outliers": outliers
    }

# Metodos de Imputacion #
# 1 .Imputacion por Media o Moda
def imputacion_media(data_list):
    print("\nCalculando media para imputación...")
    print(data_list)
    media = get_mean(data_list)
    print(f"Media calculada: {media}")
    data_imputada = [media if pd.isna(x) else x for x in data_list]
    # Creamos la lista de "auditoría" (solo los cambios)
    cambios = []
    for i, x in enumerate(data_list):
        if pd.isna(x):
            cambios.append({"indice": i, "valor_nuevo": media})
    return data_imputada,cambios

# 2.- Imputacion por vecino mas cercano (KNN)
def imputacion_knn(data_list, k=3):
    data_imputada = []
    cambios = []  # Lista para la auditoría
    
    for i, x in enumerate(data_list):
        if pd.isna(x):
            vecinos = []
            # Buscar vecinos a la izquierda
            for j in range(i-1, max(i-k-1, -1), -1):
                if not pd.isna(data_list[j]):
                    vecinos.append(data_list[j])
                if len(vecinos) >= k:
                    break
            
            # Buscar vecinos a la derecha
            for j in range(i+1, min(i+k+1, len(data_list))):
                if not pd.isna(data_list[j]):
                    vecinos.append(data_list[j])
                if len(vecinos) >= k:
                    break
            
            if vecinos:
                nuevo_valor = sum(vecinos) / len(vecinos)
                data_imputada.append(nuevo_valor)
                # Guardamos el índice, el valor nuevo y los vecinos utilizados
                cambios.append({
                    "indice": i,
                    "valor_nuevo": nuevo_valor,
                    "vecinos_usados": vecinos.copy()
                })
            else:
                data_imputada.append(x)  # Se queda como NaN si no hubo vecinos
        else:
            data_imputada.append(x)
            
    return data_imputada, cambios

# Codificacion
def codificacion_one_hot(data_list):
    categorias = sorted(set(x for x in data_list if str(x).lower() != 'nan'))
    codificacion = {cat: [1 if x == cat else 0 for x in data_list] for cat in categorias}
    return codificacion

# Normalizacion
def normalizacion_min_max(data_list):
    min_val = get_min_value(data_list)
    max_val = get_max_value(data_list)
    if max_val == min_val:
        return [0.5 for _ in data_list]  # Evitar división por cero, asignamos 0.5 a todos
    return [(x - min_val) / (max_val - min_val) if not pd.isna(x) else x for x in data_list]

# def comparar_imputacion_kde(data_original, data_imputada):

#     original = pd.Series(data_original)
#     imputada = pd.Series(data_imputada)

#     mask = original.isna()

#     valores_reales = original[~mask]
#     valores_imputados = imputada[mask]

#     print("\nCantidad de valores imputados:", mask.sum())

#     plt.figure(figsize=(8,5))

#     sns.kdeplot(valores_reales, label="Valores reales", fill=True)
#     sns.kdeplot(valores_imputados, label="Valores imputados", fill=True)

#     plt.title("Comparación KDE: Reales vs Imputados")
#     plt.xlabel("Valor")
#     plt.ylabel("Densidad")
#     plt.legend()
#     plt.show()
def comparar_imputacion_kde(data_original, data_imputada):

    original = pd.Series(data_original)
    imputada = pd.Series(data_imputada)

    valores_reales = original.dropna()
    valores_imputados = imputada.dropna()

    print("\nCantidad de valores imputados:", original.isna().sum())

    plt.figure(figsize=(8,5))

    sns.kdeplot(valores_reales, label="Valores reales", fill=True)
    sns.kdeplot(valores_imputados, label="Valores imputados", fill=True)

    plt.title("Comparación KDE: Reales vs Imputados")
    plt.xlabel("Valor")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()

def verificar_normalizacion(original, normalizada):

    original = pd.Series(original).dropna()
    normalizada = pd.Series(normalizada).dropna()

    plt.figure(figsize=(6,6))
    plt.scatter(original, normalizada)

    plt.xlabel("Valores Originales")
    plt.ylabel("Valores Normalizados")
    plt.title("Relación Original vs Normalizado")

    plt.show()

# Plantilla para pasar por cada proceso#
def validate_data(data_list,df,num):
    
    if not data_list or len(data_list) == 0:
        return "Lista vacía"
    
    if not all(isinstance(x, (int, float)) for x in data_list):
        return "Error: La columna contiene caracteres o strings."

    #Calculamos la media (asegúrate de que get_mean esté definida)
    media = get_mean(data_list)
    
    # Retornamos el diccionario de resultados
    res = {}
    if(num == 1):
        res = {
            "max": get_max_value(data_list),
            "min": get_min_value(data_list),
            "media": media,
            "desv_estandar": get_desv_stand(data_list, media),
            "atributos": get_hM_atribute(df),
            "instancias": get_hM_instance(data_list),
            "observaciones": get_observations(data_list),
            "balance_clases": get_balanceofClasses(data_list)
        }
    if(num == 2):
        res = {
        "datf": get_empty_Data(data_list),
        "iqr": calcular_estadisticas_outliers(data_list)
    }
    return res

# En este primer menu sera para la actividad 1#
def menu_operaciones(data_list,df2):
    while True:
        print("\n--- ¿Qué desea calcular? ---")
        print("1. Valor Máximo\n2. Valor Mínimo\n3. Media\n4. Desviación Estándar")
        print("5. Número de Atributos\n6. Número de Instancias\n7. Observaciones")
        print("8. Balance de Clases\n9. TODO (Resumen Completo)\n0. Volver")

        op = input("\nSeleccione una operación: ")

        if op == "0":
            break
        
        results = validate_data(data_list,df2,1)

        if isinstance(results, str):
            print(f"\n[!] {results}")
            break 

        mapeo = {
            "1": "max", "2": "min", "3": "media", 
            "4": "desv_estandar", "5": "atributos", 
            "6": "instancias", "7": "observaciones", "8": "balance_clases"
        }

        if op in mapeo:
            clave = mapeo[op]
            print(f"\n>>> {clave.replace('_', ' ').capitalize()}: {results[clave]}")
        
        elif op == "9":
            print("\n--- RESUMEN COMPLETO ---")
            for k, v in results.items():
                print(f"{k.replace('_', ' ').capitalize()}: {v}")
            # t.sleep(5)
            input("\nPresione Enter para continuar...")
        else:
            print("Opción no válida. Intente de nuevo.")

# En este segundo menu sera para la actividad 2#
def menu_operaciones2(data_list,df3):
    while True:
        print("\n--- ¿Qué desea calcular? ---")
        print("1. Datos Faltantes\n2. Datos Atipicos (IQR)\n3. Imputacion de Datos")
        print("\n0. Volver")

        op = input("\nSeleccione una operación: ")

        if op == "0":
            break
        
        results = validate_data(data_list,df3,2)

        if isinstance(results, str):
            print(f"\n[!] {results}")
            break 

        mapeo = {
            "1": "datf", "2": "iqr"
        }

        if op in mapeo:
            clave = mapeo[op]
            print(f"\n{clave.replace('_', ' ').capitalize()}: {results[clave]}")
        elif op == "3":
            data_imputada = menu_imputacion(data_list)
            print("\nDatos imputados:", data_imputada)
            if data_imputada is not None:
                comparar_imputacion_kde(data_list, data_imputada)
                op2 = input("\nDesea continuar con la codificacion?? \n1)Si\n2)No\n")
                if op2 == "1":
                    codificacion = codificacion_one_hot(data_imputada)
                    print("\nCodificación One-Hot:")
                    for cat, cod in codificacion.items():
                        print(f"{cat}: {cod}")
                    op3 = input("\nDesea continuar con la normalizacion?? \n1)Si\n2)No\n")
                    if op3 == "1":
                        normalizada = normalizacion_min_max(data_imputada)
                        verificar_normalizacion(data_imputada,normalizada)
        else:
            print("Opción no válida. Intente de nuevo.")

# Este menu es para la actividad 2, donde se muestran los metodos de imputacion, codificacion y normalizacion#
def menu_imputacion(data_list):

    data_imputada = None

    while True:
        print("\n--- Métodos de Imputación ---")
        print("1. Imputación por Media")
        print("2. Imputación por KNN")
        print("0. Volver")

        op = input("\nSeleccione un método de imputación: ")

        if op == "0":
            break
        
        elif op == "1":
            datos, cambios = imputacion_media(data_list)
            print("\nDatos imputados por media:")
            print(datos)
            data_imputada = datos

        elif op == "2":
            datos, cambios = imputacion_knn(data_list)
            print("\nDatos imputados por KNN:")
            print(datos)
            data_imputada = datos
        
        else:
            print("Opción no válida.")

    return data_imputada

def _main():
    respuesta = True
    print('\t Bienvenido al programa \n')
    while respuesta:
        try:
            val = int(input('De que manera quiere cargar sus datos :\n1.- Local\n2.- Kaggle\n'))
            if val == 1:
                location = input('Ingrese la ubicacion de su archivo: ')
                df = load_data_local(location.strip())
            if val == 2:
                # Actualmente solo funciona con B.D. en kaggle y con Internet
                typeof = input('Ingrese el tipo de archivo a guardar : ')
                web = input('Ingrese el handle de la B.D. : ')
                df = load_data_cloud_kg(typeof.strip(),web.strip())
                # load_data_cloud_kg( "StudentPerformance.csv","neurocipher/student-performance")
            else :
                print('Solo puede ingresar 1-2. Intente de nuevo :3 \n\n')
                t.sleep(wait_time)
                continue

            if df is None:
                print('\nNo se encontro la B.D. Intente de nuevo :3 \n\n')
                t.sleep(wait_time)
                continue

            # print('Tiene algo :3')
            # Menciona que quiere realizar el usuario#
            bd = True
            while bd:
                columnas = df.columns.tolist()
                
                print("\n--- Datos cargados con éxito ---")
                print("¿Qué columna desea analizar?")
                
                for i, col in enumerate(columnas, 1):
                    print(f"{i}. {col}")
                
                opcion_todos = len(columnas) + 1
                # print(f"{opcion_todos}. Analizar TODO el dataset")
                print("0. Cargar otro archivo / Salir")

                sel = input("\nSeleccione una opción: \n")

                # Lógica para procesar la selección
                if sel == "0":
                    t.sleep(wait_time)
                    bd = False
                    continue
                else:
                    sel_int = int(sel)
                    if 1<= sel_int <= len(columnas) + 1:
                        lista = df[columnas[sel_int - 1]].values.tolist()
                        preguntaActividad = int(input('Que actividad desea realizar : \n1.- Actividad 1\n2.- Actividad 2\n'))
                        if preguntaActividad == 1:
                            menu_operaciones(lista,df)
                        if preguntaActividad == 2:
                            menu_operaciones2(lista,df)
                    # else:
                    #     print('Hola')

            #Al finalizar el proceso se confirma si quiere intentar de nuevo el proceso# 
            respuesta =  int(input('\tQuiere intentarlo con otra B.D?? \n1)Si\n2)No\n'))
            respuesta = True if respuesta == 1 else False

        except Exception as e:
            print('\n Hubo un error al momento de consultar la B.D. \n '+e+'\n Intente de nuevo :3\n\n')
            t.sleep(wait_time)
            continue

_main()