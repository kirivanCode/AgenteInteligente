Sistema de Evaluación de Profesores
Descripción General
Sistema web desarrollado con Streamlit para la evaluación de profesores, que incluye análisis de sentimientos, moderación automática de comentarios y estadísticas. El sistema utiliza Firebase como base de datos y diferentes tecnologías de procesamiento de lenguaje natural.
Componentes Principales
1. Inicialización y Configuración

Utiliza Firebase Admin para la gestión de la base de datos
Implementa SpaCy para el procesamiento de lenguaje natural en español
Configura Streamlit para la interfaz de usuario

2. ModeradorComentarios (Clase Principal)
Características principales:

Análisis de sentimientos mejorado para español
Detección de lenguaje ofensivo
Clasificación mediante KNN (K-Nearest Neighbors)
Gestión de comentarios en Firebase

Métodos principales:

_analizar_sentimiento(): Análisis de sentimiento específico para español
_calcular_score_ofensivo(): Detecta y calcula nivel de contenido ofensivo
evaluar_comentario(): Método principal para procesar nuevos comentarios
obtener_estadisticas_profesor(): Genera estadísticas por profesor

3. Interfaz de Usuario (Streamlit)
Páginas:

Evaluación

Formulario para nuevos comentarios
Visualización de resultados del análisis
Retroalimentación instantánea


Estadísticas

Métricas por profesor
Visualizaciones con Plotly
Análisis de palabras frecuentes


Moderación

Panel de control para comentarios
Filtros por estado
Visualización detallada de análisis



Características Técnicas
Análisis de Sentimientos

Diccionario personalizado de palabras positivas/negativas
Integración con TextBlob
Sistema de clasificación en 5 niveles con emojis

Sistema de Moderación

Patrones de expresiones regulares para detectar contenido ofensivo
Lista de palabras prohibidas
Modelo KNN para clasificación automática
Sistema de puntuación ponderada

Procesamiento de Datos

Vectorización de texto (TF-IDF)
Escalado de características
Análisis de palabras clave con SpaCy

Requisitos Técnicos

Python 3.x
Firebase Admin SDK
Streamlit
SpaCy (modelo es_core_news_sm)
TextBlob
Pandas, NumPy
Plotly
Scikit-learn

Seguridad y Privacidad

Autenticación mediante Firebase
Identificación de estudiantes y profesores
Moderación automática de contenido
Almacenamiento seguro en Cloud Firestore

Métricas y Análisis

Score de contenido ofensivo (0-1)
Puntuación de sentimiento (-1 a 1)
Clasificación de sentimientos (muy_positivo a muy_negativo)
Estadísticas agregadas por profesor
