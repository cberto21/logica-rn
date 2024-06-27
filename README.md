# Redes Neuronales Artificiales (RNA)

## ¿Qué son las redes neuronales?

Una red neuronal es un programa, o modelo, de machine learning que toma decisiones de forma similar al cerebro humano, utilizando procesos que imitan la forma en que las neuronas biológicas trabajan juntas para identificar fenómenos, sopesar opciones y llegar a conclusiones.

<p style="float: right; margin-left: 10px;">
    <img src="https://assets.isu.pub/document-structure/210510021110-9ddcf1449571f5a351db79d67bf90513/v1/1850068f7d951490560789f46cb9c89f.jpg" alt="Imagen" width="300px">
</p>


Toda red neuronal consta de capas de nodos o neuronas artificiales: una capa de entrada, una o varias capas ocultas y una capa de salida. Cada nodo se conecta a los demás y tiene su propia ponderación y umbral asociados. Si la salida de cualquier nodo individual está por encima del valor umbral especificado, ese nodo se activa y envía datos a la siguiente capa de la red. De lo contrario, no se pasa ningún dato a la siguiente capa de la red.


 <img src="https://www.droneguru.es/wp-content/uploads/2019/11/004.png" alt="Imagen" width="600px">

Las redes neuronales se basan en datos de entrenamiento para aprender y mejorar su precisión con el tiempo. Una vez perfeccionadas, se convierten en potentes herramientas en informática e inteligencia artificial, que nos permiten clasificar y agrupar datos a gran velocidad. Las tareas de reconocimiento de voz o de imágenes pueden llevar minutos frente a horas si se comparan con la identificación manual por parte de expertos humanos. Uno de los ejemplos más conocidos de red neuronal es el algoritmo de búsqueda de Google.


<img src="https://www.xeridia.com/wp-content/uploads/2020/08/entrenar-redes-neuronales-artificiales.jpg.webp" alt="Imagen" width="600px">



Las redes neuronales a veces se denominan redes neuronales artificiales (ANN) o redes neuronales simuladas (SNN). Son un subconjunto del machine learning y el núcleo de los modelos de deep learning.

Libro electrónico: Cree flujos de trabajo de IA responsables con gobernanza de IA. Descubra los componentes básicos y las buenas prácticas para ayudar a sus equipos a acelerar la IA responsable.

## ¿Cómo funcionan las redes neuronales?

Piense en cada nodo individual como su propio modelo de regresión lineal, compuesto por datos de entrada, ponderaciones, un sesgo (o umbral) y una salida. La fórmula sería la siguiente:

∑wixi + sesgo = w1x1 + w2x2 + w3x3 + sesgo
salida = f(x) = 1 if ∑w1x1 + b>= 0; 0 if ∑w1x1 + b < 0

Una vez determinada la capa de entrada, se asignan las ponderaciones. Estas ponderaciones ayudan a determinar la importancia de cualquier variable, ya que las más grandes contribuyen de forma más significativa a la salida en comparación con otras entradas. A continuación, todas las entradas se multiplican por sus respectivas ponderaciones y se suman. Después, la salida se pasa a través de una función de activación, que determina la salida. Si esa salida supera un umbral determinado, se «dispara» (o activa) el nodo, pasando los datos a la siguiente capa de la red. Esto da como resultado que la salida de un nodo se convierta en la entrada del siguiente nodo. Este proceso de pasar datos de una capa a la siguiente capa define esta red neuronal como una red de proalimentación.

Desglosemos el aspecto de un único nodo utilizando valores binarios. Podemos aplicar este concepto a un ejemplo más tangible, como si deberías ir a hacer surf (Sí: 1, No: 0). La decisión de ir o no ir es nuestro resultado previsto, o y-hat. Supongamos que hay tres factores que influyen en tu decisión:

- ¿Las olas son buenas? (Sí: 1, No: 0)
- ¿Está el pico despejado? (Sí: 1, No: 0)
- ¿Ha habido un ataque de tiburones recientemente? (Sí: 0, No: 1)

Entonces, supongamos lo siguiente, dándonos las siguientes entradas:
- X1 = 1, ya que las olas son buenas
- X2 = 0, ya que está lleno de gente
- X3 = 1, ya que no ha habido un ataque reciente de tiburón

Ahora, tenemos que asignar algunas ponderaciones para determinar la importancia. Unas ponderaciones mayores significan que determinadas variables son más importantes para la decisión o el resultado.

- W1 = 5, ya que las grandes olas no aparecen con frecuencia
- W2 = 2, ya que estás acostumbrado a las multitudes
- W3 = 4, ya que tienes miedo a los tiburones

Por último, también supondremos un valor umbral de 3, lo que se traduciría en un valor de sesgo de –3. Con todas las entradas, podemos empezar a introducir valores en la fórmula para obtener la salida deseada.

Y-hat = (1*5) + (0*2) + (1*4) – 3 = 6

Si utilizamos la función de activación del principio de esta sección, podemos determinar que la salida de este nodo sería 1, ya que 6 es mayor que 0. En este caso, iría a surfear; pero si ajustamos las ponderaciones o el umbral, podemos obtener resultados diferentes del modelo. Cuando observamos una decisión, como en el ejemplo anterior, podemos ver cómo una red neuronal podría tomar decisiones cada vez más complejas en función de la salida de las decisiones o capas anteriores.

En el ejemplo anterior, utilizamos perceptrones para ilustrar algunas de las matemáticas que están en juego aquí, pero las redes neuronales aprovechan las neuronas sigmoidales, que se distinguen por tener valores entre 0 y 1. Dado que las redes neuronales se comportan de forma similar a los árboles de decisión, con datos en cascada de un nodo a otro, tener valores x entre 0 y 1 reducirá el impacto de cualquier cambio dado de una sola variable en la salida de cualquier nodo dado y, posteriormente, en la salida de la red neuronal.

Cuando empecemos a pensar en casos de uso más prácticos para las redes neuronales, como el reconocimiento o la clasificación de imágenes, aprovecharemos el aprendizaje supervisado, o conjuntos de datos etiquetados, para entrenar el algoritmo. Al entrenar el modelo, queremos evaluar su precisión utilizando una función de coste (o pérdida). También se conoce como error cuadrático medio (MSE). En la siguiente ecuación:

i representa el índice de la muestra,
y-hat es el resultado previsto,
y es el valor real y
m es el número de muestras.
= =1/2 ∑129_(=1)^▒( ̂^(() )−^(() ) )^2

En última instancia, el objetivo es minimizar nuestra función de coste para garantizar la corrección del ajuste para cualquier observación dada. A medida que el modelo ajusta sus ponderaciones y sesgos, utiliza la función de coste y el aprendizaje por refuerzo para alcanzar el punto de convergencia, o el mínimo local. El proceso por el que el algoritmo ajusta sus ponderaciones es el descenso gradiente, que permite al modelo determinar la dirección que debe tomar para reducir los errores (o minimizar la función de coste). Con cada ejemplo de entrenamiento, los parámetros del modelo se ajustan para converger gradualmente en el mínimo.

La mayoría de las redes neuronales profundas son alimentadas, lo que significa que fluyen en una sola dirección, de la entrada a la salida. Sin embargo, también puede entrenar su modelo mediante retropropagación; es decir, moverse en la dirección opuesta, de la salida a la entrada. La retropropagación nos permite calcular y atribuir el error asociado a cada neurona, lo que nos permite ajustar y encajar adecuadamente los parámetros del modelo o modelos.

## Estructura y componentes básicos de una red neuronal

La estructura básica de una red neuronal consta de capas de neuronas interconectadas.
- Una capa de entrada que, estrictamente hablando, no está formada por neuronas artificiales, simplemente recibe los datos de entrada.
- Capas ocultas, que reciben este nombre porque no son visibles ni desde la entrada ni desde la salida. El número de capas ocultas es variable y depende del objetivo perseguido.
- Una capa de salida que, en función del tipo de escenario en el que nos encontremos (clasificación o regresión) tendrá una o más neuronas.

La red aprende examinando los registros individuales, generando una predicción para cada registro y realizando ajustes a las ponderaciones cuando realiza una predicción incorrecta. Este proceso se repite muchas veces y la red sigue mejorando sus predicciones hasta haber alcanzado uno o varios criterios de parada.

Al principio, todas las ponderaciones son aleatorias y las respuestas que resultan de la red son, posiblemente, disparatadas. La red aprende a través del entrenamiento. Continuamente se presentan a la red ejemplos para los que se conoce el resultado, y las respuestas que proporciona se comparan con los resultados conocidos. La información procedente de esta comparación se pasa hacia atrás a través de la red, cambiando las ponderaciones gradualmente. A medida que progresa el entrenamiento, la red se va haciendo cada vez más precisa en la replicación de resultados conocidos. Una vez entrenada, la red se puede aplicar a casos futuros en los que se desconoce el resultado.


<img src="https://miro.medium.com/v2/resize:fit:1156/format:webp/1*DzPv2JB24A9Po7sblKs5EA.jpeg" alt="Imagen" width="600px">

## Aplicaciones

## Tipos de redes neuronales

Existen diferentes clasificaciones que separan las redes neuronales en torno a su número de capas, tipos de conexiones o grado de las conexiones. Vamos a intentar aclarar
