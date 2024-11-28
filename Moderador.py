import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from textblob import TextBlob
import spacy
import re
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Evaluación de Profesores",
    page_icon="👨‍🏫",
    layout="wide"
)

# Inicialización de Firebase y modelos
@st.cache_resource
def inicializar_recursos():
    if not firebase_admin._apps:
        cred = credentials.Certificate(r'C:\Users\Ivan\Documents\Python\proyecto_grado\data2.json')
        firebase_admin.initialize_app(cred)
    
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        st.error("Por favor ejecuta 'python -m spacy download es_core_news_sm' en la terminal")
        st.stop()
    
    return {
        'db': firestore.client(),
        'nlp': nlp
    }

recursos = inicializar_recursos()

class ModeradorComentarios:
    def __init__(self, recursos):
        self.db = recursos['db']
        self.nlp = recursos['nlp']
        self.patrones_ofensivos = self._cargar_patrones_ofensivos()
        self.palabras_prohibidas = self._cargar_palabras_prohibidas()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.scaler = StandardScaler()
        self.modelo_entrenado = False
        self._inicializar_modelo()
    
    def _cargar_patrones_ofensivos(self):
        return {
            r'\b(hijueputa|hp|puto|puta|putas|putos)\b': 1.0,
            r'\b(idiota|estupido|imbecil|pendejo)\b': 0.8,
            r'\b(mierda|mrd|shit)\b': 1.0,
            r'\b(incompetente|inutil|mediocre)\b': 0.7,
            r'\b(basura|porqueria)\b': 0.8,
            r'\b(malparido|gonorrea|triple)\b': 1.0,
            r'\b(perro|perra)\b': 0.7,
            r'\b(malo|pesimo|horrible)\b': 0.5
        }
    
    def _cargar_palabras_prohibidas(self):
        return ['mierda', 'hijueputa', 'hp', 'puto', 'puta', 'idiota', 'estupido', 
                'malparido', 'gonorrea', 'basura', 'imbecil', 'pendejo', 'vergas']

    def _analizar_sentimiento(self, texto):
        """Análisis de sentimiento mejorado para español"""
        # Lista de palabras positivas en español
        palabras_positivas = {
            'excelente': 1.0, 'extraordinario': 1.0, 'increíble': 0.9, 'fantástico': 0.9,
            'maravilloso': 0.9, 'genial': 0.8, 'encanta': 0.8, 'perfecto': 0.8,
            'bueno': 0.7, 'buena': 0.7, 'buenos': 0.7, 'buenas': 0.7,
            'estupendo': 0.8, 'grandioso': 0.8, 'magnífico': 0.9,
            'dedicado': 0.6, 'profesional': 0.6, 'organizado': 0.6,
            'dinámico': 0.6, 'didáctico': 0.7, 'claro': 0.6,
            'paciente': 0.7, 'puntual': 0.6, 'respetuoso': 0.7,
            'comprometido': 0.7, 'responsable': 0.7, 'actualizado': 0.6,
            'práctico': 0.6, 'excelencia': 0.9, 'brillante': 0.8,
            'motivador': 0.8, 'inspirador': 0.8, 'mejor': 0.9,
            'bien': 0.7, 'agradable': 0.7, 'interesante': 0.7,
            'recomendado': 0.8, 'excepcional': 0.9, 'eficiente': 0.7,
            'positivo': 0.7, 'satisfecho': 0.8, 'feliz': 0.8,
            'gracias': 0.7, 'agradecido': 0.8, 'útil': 0.7
        }
        
        # Lista de palabras negativas en español
        palabras_negativas = {
            'malo': -0.7, 'mala': -0.7, 'malos': -0.7, 'malas': -0.7,
            'pésimo': -0.9, 'horrible': -0.9, 'terrible': -0.9,
            'deficiente': -0.7, 'mediocre': -0.7, 'inadecuado': -0.6,
            'desorganizado': -0.6, 'confuso': -0.6, 'irresponsable': -0.7,
            'impuntual': -0.6, 'irrespetuoso': -0.7, 'desactualizado': -0.6,
            'aburrido': -0.6, 'monótono': -0.6, 'incomprensible': -0.7,
            'peor': -0.8, 'difícil': -0.5, 'complicado': -0.5,
            'insatisfecho': -0.7, 'decepcionado': -0.7, 'frustrado': -0.7,
            'ineficiente': -0.6, 'negativo': -0.6, 'mal': -0.6,
            'desagradable': -0.6, 'insuficiente': -0.6
        }
        
        palabras = texto.lower().split()
        score_total = 0
        palabras_encontradas = 0
        
        for palabra in palabras:
            if palabra in palabras_positivas:
                score_total += palabras_positivas[palabra]
                palabras_encontradas += 1
            elif palabra in palabras_negativas:
                score_total += palabras_negativas[palabra]
                palabras_encontradas += 1
        
        # Análisis básico de TextBlob como respaldo
        blob_score = TextBlob(texto).sentiment.polarity
        
        # Combinar scores
        if palabras_encontradas > 0:
            score_final = score_total / palabras_encontradas
            # Dar más peso al análisis de palabras clave que a TextBlob
            return 0.8 * score_final + 0.2 * blob_score
        else:
            return blob_score

    def _clasificar_sentimiento(self, score):
        """Clasifica el sentimiento en categorías"""
        if score > 0.5:
            return "muy_positivo"
        elif score > 0.2:
            return "positivo"
        elif score >= -0.2:
            return "neutral"
        elif score >= -0.5:
            return "negativo"
        else:
            return "muy_negativo"

    def _get_emoji_sentimiento(self, clasificacion):
        """Retorna emoji basado en la clasificación del sentimiento"""
        emojis = {
            "muy_positivo": "😄",
            "positivo": "🙂",
            "neutral": "😐",
            "negativo": "🙁",
            "muy_negativo": "😢"
        }
        return emojis.get(clasificacion, "😐")
    

    def _inicializar_modelo(self):
        try:
            comentarios_ref = self.db.collection('moderacion').stream()
            datos = [doc.to_dict() for doc in comentarios_ref]
            
            if len(datos) > 10:
                df = pd.DataFrame(datos)
                
                X_text = self.vectorizer.fit_transform(df['comentario'])
                
                caracteristicas_adicionales = np.column_stack([
                    df['score_ofensivo'],
                    df['sentiment_score']
                ])
                
                caracteristicas_adicionales_norm = self.scaler.fit_transform(caracteristicas_adicionales)
                
                X_combined = np.hstack([
                    X_text.toarray(),
                    caracteristicas_adicionales_norm
                ])
                
                y = df['es_apropiado'].astype(int)
                
                self.knn.fit(X_combined, y)
                self.modelo_entrenado = True
            else:
                self.modelo_entrenado = False
        except Exception as e:
            st.warning(f"No se pudo inicializar el modelo KNN: {str(e)}")
            self.modelo_entrenado = False
    
    def _limpiar_texto(self, texto):
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', ' ', texto)
        return texto
    
    def _contar_repeticiones_ofensivas(self, texto_limpio):
        contador = 0
        for palabra in self.palabras_prohibidas:
            contador += texto_limpio.count(palabra)
        return contador
    
    def _detectar_palabras_clave(self, texto):
        doc = self.nlp(texto)
        return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']]
    
    def _calcular_score_ofensivo(self, texto):
        texto_limpio = self._limpiar_texto(texto)
        score = 0
        
        for patron, peso in self.patrones_ofensivos.items():
            matches = len(re.findall(patron, texto_limpio))
            if matches > 0:
                score += peso * matches
        
        num_palabras_ofensivas = self._contar_repeticiones_ofensivas(texto_limpio)
        if num_palabras_ofensivas > 1:
            score *= (1 + (num_palabras_ofensivas * 0.5))
        
        score = min(max(score / 2, 0), 1)
        
        if num_palabras_ofensivas > 0:
            score = max(score, 0.7)
        
        return score
    
    def _predecir_knn(self, comentario, score_ofensivo, sentiment_score):
        try:
            if not self.modelo_entrenado:
                return None
            
            X_text = self.vectorizer.transform([comentario])
            caracteristicas_adicionales = np.array([[score_ofensivo, sentiment_score]])
            caracteristicas_adicionales_norm = self.scaler.transform(caracteristicas_adicionales)
            
            X_combined = np.hstack([
                X_text.toarray(),
                caracteristicas_adicionales_norm
            ])
            
            prediccion = self.knn.predict(X_combined)
            probabilidades = self.knn.predict_proba(X_combined)
            
            return {
                'es_apropiado': bool(prediccion[0]),
                'confianza': float(max(probabilidades[0]))
            }
        except Exception as e:
            st.warning(f"Error en predicción KNN: {str(e)}")
            return None
    
    def evaluar_comentario(self, comentario, id_profesor, id_estudiante):
        score_ofensivo = self._calcular_score_ofensivo(comentario)
        palabras_clave = self._detectar_palabras_clave(comentario)
        sentiment_score = self._analizar_sentimiento(comentario)
        
        # Clasificar el sentimiento
        clasificacion_sentimiento = self._clasificar_sentimiento(sentiment_score)
        emoji_sentimiento = self._get_emoji_sentimiento(clasificacion_sentimiento)
        
        prediccion_knn = self._predecir_knn(comentario, score_ofensivo, sentiment_score)
        
        if prediccion_knn and prediccion_knn['confianza'] > 0.7:
            es_apropiado = prediccion_knn['es_apropiado']
        else:
            es_apropiado = score_ofensivo < 0.5
        
        evaluacion = {
            'comentario': comentario,
            'fecha': datetime.now(),
            'id_profesor': id_profesor,
            'id_estudiante': id_estudiante,
            'score_ofensivo': float(score_ofensivo),
            'sentiment_score': float(sentiment_score),
            'clasificacion_sentimiento': clasificacion_sentimiento,
            'emoji_sentimiento': emoji_sentimiento,
            'palabras_clave': palabras_clave,
            'es_apropiado': es_apropiado,
            'estado': 'desaprobado' if not es_apropiado else 'aprobado',
            'timestamp': firestore.SERVER_TIMESTAMP,
            'prediccion_knn': prediccion_knn['confianza'] if prediccion_knn else None
        }
        
        self.db.collection('moderacion').add(evaluacion)
        return evaluacion
    
    def obtener_estadisticas_profesor(self, id_profesor):
        evaluaciones = self.db.collection('moderacion')\
            .where('id_profesor', '==', id_profesor)\
            .stream()
        
        evaluaciones_list = [doc.to_dict() for doc in evaluaciones]
        if not evaluaciones_list:
            return None
            
        df = pd.DataFrame(evaluaciones_list)
        
        return {
            'total_evaluaciones': len(df),
            'promedio_sentiment': df['sentiment_score'].mean(),
            'porcentaje_apropiados': (df['es_apropiado'].sum() / len(df)) * 100,
            'palabras_frecuentes': pd.Series([palabra for palabras in df['palabras_clave'] 
                                            for palabra in palabras]).value_counts().head(10).to_dict()
        }

# Inicializar el moderador
moderador = ModeradorComentarios(recursos)

# Interfaz de Streamlit
st.title("🎓 Sistema de Evaluación de Profesores")

# Sidebar para navegación
pagina = st.sidebar.selectbox(
    "Selecciona una página",
    ["Evaluación", "Estadísticas", "Moderación"]
)

if pagina == "Evaluación":
    st.header("📝 Nueva Evaluación")
    
    with st.form("formulario_evaluacion"):
        id_profesor = st.text_input("ID del Profesor")
        id_estudiante = st.text_input("ID del Estudiante")
        comentario = st.text_area("Comentario")
        
        submitted = st.form_submit_button("Enviar Evaluación")
        
        if submitted and comentario and id_profesor and id_estudiante:
            resultado = moderador.evaluar_comentario(comentario, id_profesor, id_estudiante)
            
            if resultado['es_apropiado']:
                st.success("¡Evaluación enviada correctamente!")
            else:
                st.error("El comentario ha sido desaprobado por contener lenguaje inapropiado.")
                st.warning("Por favor, modifica tu comentario y evita usar lenguaje ofensivo.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Score Ofensivo", 
                    f"{resultado['score_ofensivo']:.2f}",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Sentimiento", 
                    f"{resultado['sentiment_score']:.2f}"
                )
                st.markdown(f"### {resultado['emoji_sentimiento']} {resultado['clasificacion_sentimiento'].replace('_', ' ').title()}")
            with col3:
                estado_color = "red" if resultado['estado'] == "desaprobado" else "green"
                st.markdown(
                    f"<h3 style='text-align: center; color: {estado_color};'>{resultado['estado'].upper()}</h3>",
                    unsafe_allow_html=True
                )
            
            if resultado.get('prediccion_knn'):
                st.markdown("---")
                st.markdown("### Análisis del Modelo de Aprendizaje")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Confianza del Modelo", 
                        f"{resultado['prediccion_knn']:.2%}"
                    )
                with col2:
                    st.markdown(
                        f"El modelo está {'muy seguro' if resultado['prediccion_knn'] > 0.8 else 'moderadamente seguro'} "
                        f"de su predicción"
                    )
            
            if not resultado['es_apropiado']:
                st.markdown("---")
                st.markdown("### Sugerencias para mejorar el comentario:")
                st.markdown("""
                - Evita usar palabras ofensivas o insultos
                - Céntrate en aspectos constructivos
                - Describe situaciones específicas
                - Usa un lenguaje respetuoso y profesional
                """)

elif pagina == "Estadísticas":
    st.header("📊 Estadísticas por Profesor")
    
    id_profesor = st.text_input("Ingrese ID del Profesor para ver estadísticas")
    if id_profesor:
        stats = moderador.obtener_estadisticas_profesor(id_profesor)
        
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total de Evaluaciones", stats['total_evaluaciones'])
                st.metric("Promedio de Sentimiento", f"{stats['promedio_sentiment']:.2f}")
            
            with col2:
                st.metric("Porcentaje de Comentarios Apropiados", 
                         f"{stats['porcentaje_apropiados']:.1f}%")
            
            if stats['palabras_frecuentes']:
                fig = px.bar(
                    x=list(stats['palabras_frecuentes'].keys()),
                    y=list(stats['palabras_frecuentes'].values()),
                    title="Palabras más frecuentes en las evaluaciones"
                )
                st.plotly_chart(fig)
        else:
            st.warning("No se encontraron estadísticas para este profesor")

elif pagina == "Estadísticas":
    st.header("📊 Estadísticas por Profesor")
    
    id_profesor = st.text_input("Ingrese ID del Profesor para ver estadísticas")
    if id_profesor:
        stats = moderador.obtener_estadisticas_profesor(id_profesor)
        
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total de Evaluaciones", stats['total_evaluaciones'])
                st.metric("Promedio de Sentimiento", f"{stats['promedio_sentiment']:.2f}")
            
            with col2:
                st.metric("Porcentaje de Comentarios Apropiados", 
                         f"{stats['porcentaje_apropiados']:.1f}%")
            
            if stats['palabras_frecuentes']:
                fig = px.bar(
                    x=list(stats['palabras_frecuentes'].keys()),
                    y=list(stats['palabras_frecuentes'].values()),
                    title="Palabras más frecuentes en las evaluaciones"
                )
                st.plotly_chart(fig)
        else:
            st.warning("No se encontraron estadísticas para este profesor")

elif pagina == "Moderación":
    st.header("⚖️ Panel de Moderación")
    
    estado_filtro = st.selectbox(
        "Filtrar por estado",
        ["desaprobado", "aprobado"]
    )
    
    comentarios_filtrados = recursos['db'].collection('moderacion')\
        .where('estado', '==', estado_filtro)\
        .stream()
    
    comentarios_list = [doc.to_dict() for doc in comentarios_filtrados]
    
    if comentarios_list:
        st.write(f"### Comentarios {estado_filtro.title()}")
        
        for comentario in comentarios_list:
            with st.expander(f"Comentario de {comentario['id_estudiante']} para {comentario['id_profesor']}"):
                st.write("**Comentario:**", comentario['comentario'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score Ofensivo", f"{comentario['score_ofensivo']:.2f}")
                with col2:
                    st.metric("Sentimiento", f"{comentario['sentiment_score']:.2f}")
                with col3:
                    if 'clasificacion_sentimiento' in comentario:
                        st.markdown(f"### {comentario['emoji_sentimiento']} {comentario['clasificacion_sentimiento'].replace('_', ' ').title()}")
                
                st.write("**Palabras clave:**", ", ".join(comentario['palabras_clave']))
                
                if 'prediccion_knn' in comentario and comentario['prediccion_knn'] is not None:
                    st.metric("Confianza del Modelo", f"{comentario['prediccion_knn']:.2%}")
    else:
        st.info(f"No hay comentarios en estado: {estado_filtro}")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado con ❤️ por UTS")
