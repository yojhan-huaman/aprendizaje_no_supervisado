from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

df_global = None

# ---------- Database Setup ----------
def crear_base():
    conn = sqlite3.connect('temperaturas.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS temperaturas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            valor REAL,
            anomalia TEXT
        )''')
    c.execute('''CREATE TABLE IF NOT EXISTS evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            valor REAL,
            anomalia TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    conn.commit()
    conn.close()

# ---------- Save to Database ----------
def guardar_en_bd(df):
    conn = sqlite3.connect('temperaturas.db')
    c = conn.cursor()
    for _, fila in df.iterrows():
        c.execute("INSERT INTO temperaturas (valor, anomalia) VALUES (?, ?)", 
                  (fila['Temperatura (°C)'], fila['Anomalia']))
    conn.commit()
    conn.close()

def guardar_evaluacion(valor, anomalia):
    conn = sqlite3.connect('temperaturas.db')
    c = conn.cursor()
    c.execute("INSERT INTO evaluaciones (valor, anomalia) VALUES (?, ?)", (valor, anomalia))
    conn.commit()  # Confirmar la inserción
    conn.close()

# ---------- Graph Creation ----------
def crear_grafico(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df['Temperatura (°C)'], label='Temperatura')
    plt.scatter(df.index[df['Anomalia'] == 'Si'],
                df['Temperatura (°C)'][df['Anomalia'] == 'Si'],
                color='red', label='Anomalía')

def crear_grafico_con_nuevo_dato(df, nuevo_valor, es_anomalia):
    plt.figure(figsize=(10, 4))
    plt.plot(df['Temperatura (°C)'], label='Temperatura')

    plt.scatter(df.index[df['Anomalia'] == 'Si'],
                df['Temperatura (°C)'][df['Anomalia'] == 'Si'],
                color='red', label='Anomalía')

    nuevo_indice = len(df)
    color_nuevo = 'purple' if es_anomalia == 'Si' else 'blue'
    plt.scatter(nuevo_indice, nuevo_valor, color=color_nuevo, label='Nuevo Dato')

    plt.axvline(x=nuevo_indice - 0.5, color='gray', linestyle='--', label='Evaluación')
    plt.legend()
    plt.title("Evaluación de Nuevo Dato")
    plt.xlabel("Índice")
    plt.ylabel("Temperatura (°C)")
    plt.tight_layout()

    # Eliminar la imagen anterior si existe
    image_path = 'static/grafico.png'
    if os.path.exists(image_path):
        os.remove(image_path)

    # Generar y guardar la nueva imagen con el mismo nombre
    os.makedirs('static', exist_ok=True)
    plt.savefig(image_path)
    plt.close()

# ---------- Initialize App ----------
crear_base()

@app.template_global()
def current_time():
    return datetime.now()

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    global df_global

    num_datos = 100
    porcentaje_anomalias = float(request.form.get('porcentaje', 5))

    if request.method == 'POST':
        if request.form.get('accion') == 'cargar':
            # Cargar archivo pero NO procesar aún
            archivo = request.files.get('csv_file')
            if archivo and archivo.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(archivo)
                    if 'Temperatura (°C)' not in df.columns:
                        columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
                        if len(columnas_numericas) == 1:
                            df.rename(columns={columnas_numericas[0]: 'Temperatura (°C)'}, inplace=True)
                        else:
                            return render_template("index.html", error="No se pudo identificar columna de temperatura.", cantidad=100, porcentaje=porcentaje_anomalias)
                    
                    df_global = df.copy()
                    return render_template("index.html", 
                                           tabla=df.to_html(classes='table table-bordered'),
                                           cantidad=len(df), 
                                           porcentaje=porcentaje_anomalias)
                except Exception as e:
                    return render_template("index.html", error=f"Error al cargar el CSV: {e}", cantidad=100, porcentaje=porcentaje_anomalias)

        elif request.form.get('procesar_csv') == '1':
            if df_global is not None:
                model = IsolationForest(contamination=porcentaje_anomalias / 100, random_state=42)
                model.fit(df_global[['Temperatura (°C)']])
                df_global['Anomalia'] = model.predict(df_global[['Temperatura (°C)']])
                df_global['Anomalia'] = df_global['Anomalia'].apply(lambda x: 'Si' if x == -1 else 'No')

                guardar_en_bd(df_global)
                crear_grafico(df_global)

                return render_template("index.html", 
                                       tabla=df_global.to_html(classes='table table-bordered'),
                                       cantidad=len(df_global), 
                                       porcentaje=porcentaje_anomalias)

        else:
            # Generar datos sintéticos si no hay CSV
            num_datos = int(request.form['cantidad'])
            porcentaje_anomalias = float(request.form['porcentaje'])

            normales = int(num_datos * (1 - porcentaje_anomalias / 100))
            anomalias = num_datos - normales

            temps_normales = np.random.normal(loc=22.0, scale=0.5, size=normales)
            temps_anomalias = np.random.uniform(low=5.0, high=40.0, size=anomalias)
            temperaturas = np.concatenate([temps_normales, temps_anomalias])
            np.random.shuffle(temperaturas)

            df = pd.DataFrame({'Temperatura (°C)': temperaturas})

            model = IsolationForest(contamination=porcentaje_anomalias / 100, random_state=42)
            model.fit(df[['Temperatura (°C)']])
            df['Anomalia'] = model.predict(df[['Temperatura (°C)']])
            df['Anomalia'] = df['Anomalia'].apply(lambda x: 'Si' if x == -1 else 'No')

            df_global = df.copy()
            guardar_en_bd(df)
            crear_grafico(df)

            return render_template("index.html", 
                                   tabla=df.to_html(classes='table table-bordered'),
                                   cantidad=num_datos, 
                                   porcentaje=porcentaje_anomalias)

    return render_template("index.html", tabla=None, cantidad=num_datos, porcentaje=porcentaje_anomalias)

def cargar_csv():
    global df_global
    archivo = request.files['csv_file']
    
    if archivo.filename == '':
        return redirect(url_for('index'))

    # Verificar que el archivo sea CSV
    if not archivo.filename.endswith('.csv'):
        return render_template("index.html", error="Por favor, sube un archivo CSV válido.", cantidad=100, porcentaje=5)

    try:
        # Cargar el CSV en un DataFrame
        df = pd.read_csv(archivo)
        
        # Verificar que el archivo tenga la columna 'Temperatura (°C)'
        if 'Temperatura (°C)' not in df.columns:
            columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
            if len(columnas_numericas) == 1:
                # Si el archivo tiene solo una columna numérica, renombrarla
                df.rename(columns={columnas_numericas[0]: 'Temperatura (°C)'}, inplace=True)
            else:
                # Si el CSV tiene más de una columna numérica, se considera inválido
                return render_template("index.html", error="CSV inválido, no se pudo identificar la columna de temperatura.", cantidad=100, porcentaje=5)

        porcentaje_anomalias = float(request.form.get('porcentaje', 5))
        
        # Ajustar el modelo IsolationForest
        model = IsolationForest(contamination=porcentaje_anomalias / 100, random_state=42)
        model.fit(df[['Temperatura (°C)']])
        
        # Clasificar las anomalías
        df['Anomalia'] = model.predict(df[['Temperatura (°C)']])
        df['Anomalia'] = df['Anomalia'].apply(lambda x: 'Si' if x == -1 else 'No')

        df_global = df.copy()
        guardar_en_bd(df)
        crear_grafico(df)

        return render_template("index.html", 
                               tabla=df.to_html(classes='table table-bordered'),
                               cantidad=len(df),
                               porcentaje=porcentaje_anomalias)
    
    except Exception as e:
        # En caso de error al cargar el CSV
        return render_template("index.html", error=f"Error al cargar el CSV: {e}", cantidad=100, porcentaje=5)

@app.route('/evaluar', methods=['POST'])
def evaluar_dato():
    global df_global
    try:
        valor = float(request.form['valor'])

        if df_global is None or len(df_global) == 0:
            return redirect(url_for('prediccion', error='No hay datos disponibles para evaluación.'))

        media = df_global['Temperatura (°C)'].mean()
        desviacion_estandar = df_global['Temperatura (°C)'].std()

        umbral_anomalia = 2
        es_anomalia = 'Si' if abs(valor - media) > umbral_anomalia * desviacion_estandar else 'No'

        # Guardar la evaluación en la base de datos
        guardar_evaluacion(valor, es_anomalia)

        # Crear el gráfico con el nuevo dato
        if df_global is not None:
            crear_grafico_con_nuevo_dato(df_global, valor, es_anomalia)

        return redirect(url_for('prediccion', 
                                valor=valor, 
                                es_anomalia=es_anomalia, 
                                grafico_url=url_for('static', filename='grafico.png')))

    except Exception as e:
        return redirect(url_for('prediccion', error=f"Error: {str(e)}"))

@app.route('/descargar_csv')
def descargar_csv():
    global df_global
    if df_global is not None:
        ruta_csv = 'static/datos_descargados.csv'
        df_global.to_csv(ruta_csv, index=False)
        return send_file(ruta_csv, as_attachment=True)
    else:
        return "No hay datos disponibles para descargar.", 400

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    global df_global
    valor = None
    es_anomalia = "No"
    grafico_url = None
    error = None

    if request.method == 'POST':
        try:
            valor = float(request.form['valor'])
            
            if df_global is not None and len(df_global) > 0:
                # Fit the model (ensure the model is trained or previously fit)
                model = IsolationForest(contamination=0.05, random_state=42)
                model.fit(df_global[['Temperatura (°C)']])

                # Predict anomaly
                prediction = model.predict([[valor]])
                es_anomalia = 'Sí' if prediction == -1 else 'No'

                # Create the graph with the new value
                crear_grafico_con_nuevo_dato(df_global, valor, es_anomalia)
                grafico_url = url_for('static', filename='grafico.png')
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('prediccion.html', 
                           valor=valor, 
                           es_anomalia=es_anomalia, 
                           grafico_url=grafico_url, 
                           error=error)

@app.route('/evaluaciones')
def evaluaciones():
    conn = sqlite3.connect('temperaturas.db')
    c = conn.cursor()
    c.execute('SELECT * FROM evaluaciones ORDER BY fecha DESC')
    evaluaciones = c.fetchall()
    conn.close()

    # Generar la tabla HTML para las evaluaciones
    tabla_html = '<table class="table table-bordered"><thead><tr><th>Valor</th><th>Anomalía</th><th>Fecha</th></tr></thead><tbody>'
    for evaluacion in evaluaciones:
        tabla_html += f'<tr><td>{evaluacion[1]}</td><td>{evaluacion[2]}</td><td>{evaluacion[3]}</td></tr>'
    tabla_html += '</tbody></table>'

    return render_template('evaluaciones.html', tabla=tabla_html)

@app.errorhandler(404)
def pagina_no_encontrada(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def error_interno(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)