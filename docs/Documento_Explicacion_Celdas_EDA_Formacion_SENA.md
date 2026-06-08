# Explicación de cada celda en EDA_Formacion_SENA.ipynb

Este documento describe qué hace cada celda del notebook `EDA_Formacion_SENA.ipynb`.

1. Celda 1 (Markdown)
   - Título y presentación del proyecto.
   - Describe el dataset, el objetivo del análisis, las dimensiones y el tipo de problema.

2. Celda 2 (Markdown)
   - Título de la sección de importaciones y configuración.

3. Celda 3 (Python)
   - Instala las librerías necesarias con `%pip install`.
   - Asegura que `pandas`, `numpy`, `matplotlib`, `seaborn` y `scipy` estén disponibles para el notebook.

4. Celda 4 (Python)
   - Importa librerías de análisis y visualización: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy` y `Patch` de `matplotlib`.
   - Configura advertencias para ignorarlas y define tema y tamaño de gráficos.
   - Define colores y paleta para las visualizaciones.
   - Establece la ruta del CSV.

5. Celda 5 (Markdown)
   - Título de la sección de inspección inicial y estadísticas descriptivas.

6. Celda 6 (Python)
   - Carga el dataset desde `dataset_formacion_sena.csv`.
   - Convierte columnas de fecha a tipo datetime.
   - Imprime dimensiones del dataset, periodo de fechas y niveles de formación.
   - Muestra las primeras 5 filas.

7. Celda 7 (Python)
   - Selecciona columnas numéricas.
   - Calcula y muestra estadísticas descriptivas básicas para estas variables.

8. Celda 8 (Python)
   - Recorre columnas de tipo objeto/string.
   - Muestra la cantidad de valores únicos y la frecuencia de cada valor.

9. Celda 9 (Markdown)
   - Título de la sección de análisis de valores faltantes.

10. Celda 10 (Python)
    - Calcula la cantidad y el porcentaje de valores faltantes por columna.
    - Si no hay valores faltantes, imprime mensajes de tratamiento previo.
    - Si hay valores faltantes, muestra la tabla y grafica un barplot con porcentajes.

11. Celda 11 (Markdown)
    - Título de la sección de distribución de la variable objetivo.

12. Celda 12 (Python)
    - Crea una variable binaria `ALTA_DESERCION` según un umbral de deserción del 30%.
    - Grafica histogramas, diagramas de caja y pastel para analizar la deserción.
    - Imprime el balance de clases de alta y baja deserción.

13. Celda 13 (Markdown)
    - Título de la sección de análisis univariado.

14. Celda 14 (Python)
    - Define variables numéricas específicas a analizar.
    - Grafica histogramas de estas variables y muestra su media en cada gráfico.

15. Celda 15 (Python)
    - Calcula outliers usando el criterio IQR para cada variable numérica.
    - Imprime la cantidad y el porcentaje de outliers por variable.

16. Celda 16 (Python)
    - Define variables categóricas para análisis.
    - Crea una columna extra a partir de `AÑO_APERTURA`.
    - Grafica la distribución de variables categóricas con barras horizontales.

17. Celda 17 (Python)
    - Agrupa el dataset por año y trimestre de apertura.
    - Grafica el número de fichas abiertas por periodo.

18. Celda 18 (Markdown)
    - Título de la sección de análisis bivariado.

19. Celda 19 (Python)
    - Calcula la matriz de correlaciones para variables numéricas seleccionadas.
    - Muestra un heatmap de correlación y las correlaciones con `TASA_DESERCION`.

20. Celda 20 (Python)
    - Grafica diagramas de dispersión entre variables seleccionadas y la tasa de deserción.
    - Calcula y muestra la recta de regresión lineal para cada par.

21. Celda 21 (Python)
    - Compara la deserción entre niveles, jornadas y trimestres.
    - Crea boxplots para cada variable categórica relacionada con la deserción.

22. Celda 22 (Python)
    - Analiza los 15 programas con más fichas.
    - Grafica la deserción promedio de esos programas y marca el umbral del 30%.

23. Celda 23 (Python)
    - Muestra la tendencia de fichas y deserción por trimestre.
    - Usa doble eje para comparar volumen de fichas y tasa de deserción.

24. Celda 24 (Markdown)
    - Título de la sección de preparación para modelado.

25. Celda 25 (Python)
    - Crea una copia del dataset y codifica variables categóricas como numéricas.
    - Prepara la lista de features a revisar para modelado.
    - Calcula la correlación de cada feature con la variable objetivo.
    - Grafica esas correlaciones.

26. Celda 26 (Python)
    - Detecta posibles variables redundantes con correlación mayor a 0.85.
    - Imprime pares de variables que podrían ser redundantes.

27. Celda 27 (Markdown)
    - Título de la sección de hallazgos y conclusiones.

28. Celda 28 (Python)
    - Resume los principales hallazgos del EDA.
    - Enumera las variables con mayor relación al target, el balance de clases, transformaciones sugeridas y posibles problemas de leak.
    - Identifica conclusiones de contexto sobre deserción por nivel y trimestre.
