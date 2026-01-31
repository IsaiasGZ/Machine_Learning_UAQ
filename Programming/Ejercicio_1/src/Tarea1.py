# En este proyecto realizamos algunos calculos estadisticos con Base de Datos Estructurados#
# Primero realizamos la importacion de las dependencias para que funcione#
import numpy as np
import pandas as pd
import math as m
import time as t

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
    mean = 0
    for p in values:
        if isinstance(p,(int,float)):
            mean += p
    mean = mean/len(values)
    return mean


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

# Plantilla para pasar por cada proceso#
def validate_data(data_list,df):
    
    if not data_list or len(data_list) == 0:
        return "Lista vacía"
    
    if not all(isinstance(x, (int, float)) for x in data_list):
        return "Error: La columna contiene caracteres o strings."

    #Calculamos la media (asegúrate de que get_mean esté definida)
    media = get_mean(data_list)
    
    # Retornamos el diccionario de resultados
    return {
        "max": get_max_value(data_list),
        "min": get_min_value(data_list),
        "media": media,
        "desv_estandar": get_desv_stand(data_list, media),
        "atributos": get_hM_atribute(df),
        "instancias": get_hM_instance(data_list),
        "observaciones": get_observations(data_list),
        "balance_clases": get_balanceofClasses(data_list)
    }

def menu_operaciones(data_list,df2):
    while True:
        print("\n--- ¿Qué desea calcular? ---")
        print("1. Valor Máximo\n2. Valor Mínimo\n3. Media\n4. Desviación Estándar")
        print("5. Número de Atributos\n6. Número de Instancias\n7. Observaciones")
        print("8. Balance de Clases\n9. TODO (Resumen Completo)\n0. Volver")

        op = input("\nSeleccione una operación: ")

        if op == "0":
            break
        
        results = validate_data(data_list,df2)

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
                        menu_operaciones(lista,df)
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