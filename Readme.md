# Modelo RNN-LSTM
## Detalle de los pasos seguidos
### Lectura de los datos
Los datos para el problema se dividen en tres conjuntos: train, validation y test.
A los fines prácticos de poder realizar el ajuste de hiperparámetros en un lapso de tiempo acorde a la disponibilidad de los recursos de cálculo disponibles, se programó la opción de tomar una muestra aleatoria del 10% de los conjuntos train y validation; siempre procurando mantener la distribución de clases del conjunto original.
Se grafican las 10 categorías con mayor frecuencia y las 10 con menor frecuencia, para cada uno de los conjuntos de datos. Como puede observarse, las muestras seleccionadas mantienen la distribución de los conjuntos originales; y se asegura que dentro de cada conjunto estén las 632 categorías. Es importante también resaltar que los 2 conjuntos están muy desbalanceados.
![graficos clases](https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/images/graficos_clases_2.png?raw=true)
Luego, el entrenamiento del modelo seleccionado como definitivo será realizado con todos los datos del conjunto train y se analizará el desempeño sobre todo el conjunto test.

### Creación de las clases dataset
Se crearon dos clases dataset, una para levantar todos los datos del lote y otra que lo hace en forma iterable, según las posibilidades de cómputo.

### Preprocesamiento de los títulos de las publicaciones
El preprocesamiento de los títulos se realizó utilizando las librerías Gensim y NLTK. Las tareas se ejecutan en el siguiente orden:
- Transformar todas las cadenas en minúsculas
- Eliminar etiquetas de código del tipo <i></i>, <b></b>
- Separar por un espacio de cadenas alfanuméricas
- Reemplazar signos de puntuación ASCII por espacios
- Eliminar cualquier otro caracter que no sea letras o números
- Remover espacios múltiples
- Eliminar dígitos numéricos
- Descartar las cadenas de longitud menor a 3

Una vez generado el diccionario de palabras, se eliminan de este las palabras vacías (o stopwords) del listado predefinido para español en la librería NLTK. Esto es para propiciar que en diccionario aparezcan palabras que puedan aportar información relevante.
Luego, se incluyen dos tokens especiales. Uno para las palabras desconocidas (1) y otro para el relleno al ajustar el tamaño de las cadenas (0).
Por último, se codifican las categorías con un índice, por orden de aparición. En este caso se cuenta con 632 categorías diferentes.

### PadSequences y Dataloaders
Se creó una clase PadSequences para iguales el tamaño de los datos con los que será alimentada la red.
Además, se utilizaron los dataloaders de Pytorch para pasar los datos por lotes a la red.

### Modelo RNN-LSTM baseline
Se diseñó un modelo simple con una capa de embeddings, luego una capa oculta con función de activación relu y la capa de salida. La función de pérdida utilizada para todo el trabajo fue CrossEntropyLoss, apropiada para problemas de clasificación multiclase. Además, se optó por utilizar Adam como algoritmo de optimización.
Por una cuestión de capacidad de procesamiento todos los modelos fueron entrenados en 5 épocas. La métrica utilizada para evaluar los modelos fue balanced accuracy.
El modelo baseline alcanzó un resultado de 56.4%, y fue la línea base para comprar con otras arquitecturas de red e hiperparámetros.

### Modelo RNN-LSTM para ajustar hiperparámetros
Para la búsqueda de los mejores hiperparámetro se agregó una capa lineal oculta adicional y dropout a la red. Además, se definió una función para el entrenamiento y evaluación de los modelos que recibe como parámetros el tamaño de la capa oculta, la proporción para el dropout, la función de activación, el algoritmo de optimización, tasa de aprendizaje, parámetro de regularización, épocas y la opción de guardar los parámetros del modelo entrenado.

## Experimentos
Todos los experimentos fueron registrados con MLflow para poder comprar los modelos y obtener las gráficas de pérdida para entrenamiento y evaluación.
Dada la elección de posibles hiperparámetros, había que evaluar 36 modelos. Para evitar el riesgo de perder el proceso en algún punto intermedio, se dividió en 3 etapas, es decir de a 12 modelos por vez. Además, todos los modelos fueron entrenados tomando la muestra del 10% de los datos, como se explicó en [lectura de los datos](#lectura-de-los-datos).
La funcion de activación utilizada en cada etapa fue diferente (relu, mish, tanh), mientras que los parametros hidden_dim, bdirectional, p_dropouts, opts, lrs, wds se iteraron dentro de determinados valores iguales para cada etapa.
A continuación se puede ver la tabla de registros de MLflow con los primeros resultados, ordenados por la métrica.
![registros mlflows rnn](https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/images/mlflow_rnn.png?raw=true)
Como resultado de las 36 pruebas, se obtuvieron modelos con métricas en el rango de 63.0% a 79.8%. En general, se puede apreciar que el rango de métricas de las pruebas es superior a los resultados de las pruebas de parámetros del modelo de red MLP realizada en el trabajo anterior (36.0% a 67.7%), mostrando una considerable mejora con redes recurrentes. Además, es claro que en las primeras 18 posiciones se ubican las redes con LSTM bidireccional, resultando en un parámetro clave para mejorar el desempeño del modelo.

### Mejor modelo
El modelo seleccionado como definitivo es el que puede verse seleccionado en la imagen anterior. Su elección se debió a que era el mejor modelo en términos de overfitting. Si bien no se ubica entre los primeros lugares, la diferencia en la métrica no era significativa respecto a los primeros, que mostraron overfitting incluso antes de la 3ra época. Para ejemplificar esto, se muestra en la siguiente imagen las curvas de función de pérdida para los primeros 3 modelos de la tabla.
![loss sobreajuste rnn](https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/images/loss_sobreajuste_rnn.png?raw=true)

En cambio, las curvas de pérdida para el modelo seleccionado muestran una mejor situación:
![curvas mejor modelo rnn](https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/images/loss_mejor_modelo_rnn.png?raw=true)

La métrica obtenida en el entrenamiento y evaluación con la muestra de datos fue 75.9%. Los parámetros del modelo seleccionado son:
![param mejor modelo rnn](https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/images/param_mejor_modelo_rnn.png?raw=true)

Luego, se realizó el entrenamiento y evaluación de este modelo pero con el total de los datos de train y validation. Así, la precisión balanceada ascendió a 82.1%. Este experimento también fue monitoreado con MLflow para que sea incluido en los registros. Se guardaron los parámetros entrenados del modelo para utilizarlos en el siguiente paso y evaluar el desempeño sobre el conjunto de prueba. Se incluye la gráfica de la función de pérdida para el modelo definitivo entrenado, donde se evidencia más claramente que no existe sobreajuste en el rango de épocas de entrenamiento.
![loss modelo rnn entrenado](https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/images/loss_modelo_rnn_entrenado.png?raw=true)

## Resultados y performance sobre el conjunto Test
El conjunto test fue sometido al mismo preprocesamiento utilizado en las etapas anteriores, teniendo en consideración que el diccionario es el formado a partir de los conjuntos de entrenamiento y evaluación porque el conjunto de prueba son datos "nuevos" que se presentan al modelo.
Finalmente, la métrica lograda a partir de las predicciones sobre el conjunto de prueba (89.3%) fue mejor que la obtenida con el modelo MLP (79.1%) y sumado a esto, el método RNN es bueno para el procesamiento de secuencias, por lo que se concluye que estas redes neuronales son ideales para este tipo de problemas.

## Archivos respaldo
<a href="https://github.com/RodrigoHRuiz/Diplo2022_Grupo16/blob/main/DeepLearning/02%20Modelo%20RNN-LSTM/TP%20-%20Deep%20Learning%20-%20RNN-LSTM.ipynb" target="_blank">Notebook de todo el proceso del modelo RNN-LSTM</a>
<br>
<a href="https://drive.google.com/file/d/1-m8hQhn89ac_MCwIyvfQC01RSOMbLeoo/view?usp=share_link" target="_blank">Experimentos MLflow con modelo RNN-LSTM</a>
<br>
<a href="https://drive.google.com/file/d/1XvWke8FH49BFJxkWE3LXulUhWonTT6-a/view?usp=share_link" target="_blank">Parámetros entrenados del mejor modelo RNN-LSTM</a>
