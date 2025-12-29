# Sistema RAG

> Una breve descripción de una o dos frases sobre qué hace tu sistema RAG (ej. "Sistema de Recuperación Aumentada para consultar documentos legales usando LangChain y OpenAI").

##  Tabla de Contenidos
1. [Descripción](#descripción)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [Estrutura](#estructura)

---

##  Descripción

Este proyecto implementa un flujo de trabajo **RAG (Retrieval-Augmented Generation)**. Permite cargar documentos (PDF, TXT, etc.), generar embeddings y realizar consultas en lenguaje natural para obtener respuestas basadas en el contexto de esos documentos.

**Tecnologías clave:**
* Python
* [Librería principal:LangChain / LlamaIndex]
* [Base de datos vectorial: ChromaDB ]
* [Modelo LLM, ej: phi3:mini ]

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema:

* **Anaconda** o **Miniconda**: Necesario para gestionar el entorno virtual desde el archivo `.yml`.

---

## Configuración del Entorno

Para ejecutar este proyecto, necesitas crear un entorno virtual con las dependencias exactas listadas en `environment_python.yml`.

### 1. Crear el entorno
Ejecuta el siguiente comando para instalar todas las dependencias:

```bash
conda env create -n rag_env -f environment_python.yml
```

### 2. Activar el entorno
Una vez terminada la instalación, activa el entorno:


```bash
conda activate rag_env
```

## Estructura del Systema RAG

A continuación se describe la función de los ficheros principales:

### Configuración y Utilidades
* **`config.py`**: Archivo central de configuración. Aquí se definen los modelos a utilizar, las rutas etc.
* **`url.py`**: Lista con las URLs a consultar para extraer información para la base de datos.

### Procesamiento de Datos (Ingestión)
* **`get_docs.py`**: Script encargado de la obtención de documentos a partir de URLs.
* **`preproces_data.py`**: Contiene funcioens como analizar_fuentes() que permiten analizar un documento individual para buscar formatos especificos en el documento y poder hacer un chunking de mayor calidad.
* **`preload_database.py`**: **Script crítico**. Ejecuta este archivo para generar los embeddings de los documentos en `docs/` y guardarlos en `chroma_db`.

### Ejecución y Consultas
* **`main.py`**: Punto de entrada principal de la aplicación. Interfaz par realizar consultas al Sistema RAG.
* **`query.py`**: Contiene la lógica específica para realizar la búsqueda de similitud en ChromaDB y enviar el prompt al LLM.

### Directorios
* **`chroma_db/`**: Almacenamiento persistente de la base de datos vectorial (creado automáticamente tras la carga).
* **`docs/`**: Carpeta donde debes colocar los archivos fuente (PDF, TXT, etc.) para ser procesados.
* **`evaluacion/`**: Documentos sintéticos configurados para una posible optimización del sistema mediante finetunning (NO IMPLEMENTADO)