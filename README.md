# Sistema ReciclarIA: Visión por Computador + RAG para Reciclaje

> Sistema que combina Visión por Computador para clasificar residuos y un sistema RAG (Retrieval-Augmented Generation) para proporcionar consejos precisos de reciclaje basados en documentación experta.

## Tabla de Contenidos
1. [Descripción](#descripción)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [Uso del Sistema](#uso-del-sistema)
5. [Estructura](#estructura)

---

## Descripción

Este proyecto implementa un sistema híbrido que utiliza **Visión por Computador** para clasificar imágenes de residuos en categorías como Cartón, Orgánico, Vidrio, Metal, etc., y un flujo de trabajo **RAG (Retrieval-Augmented Generation)** para consultar documentación experta sobre reciclaje y proporcionar respuestas precisas.

**Tecnologías clave:**
* Python
* Visión por Computador: ONNX Runtime con modelo entrenado en MATLAB
* RAG: LangChain con ChromaDB y Ollama
* Modelos LLM: phi3:mini (via Ollama)
* Embeddings: nomic-embed-text (via Ollama)

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema:

* **Anaconda** o **Miniconda**: Necesario para gestionar el entorno virtual desde el archivo `environment.yml`.
* **Ollama**: Plataforma para ejecutar modelos de lenguaje localmente. Descárgalo desde [ollama.ai](https://ollama.ai) e instálalo.

---

## Instalación y Configuración

### 1. Clonar o descargar el repositorio
Asegúrate de tener el código fuente en tu máquina local.

### 2. Configurar el entorno virtual
Ejecuta el siguiente comando para crear el entorno con todas las dependencias:

```bash
conda env create -f environment.yml
```

### 3. Activar el entorno
Una vez terminada la instalación, activa el entorno:

```bash
conda activate reciclaria_env
```

### 4. Instalar modelos de Ollama
El sistema requiere modelos específicos de Ollama preinstalados. Ejecuta los siguientes comandos para descargarlos:

```bash
# Modelo de lenguaje para respuestas
ollama pull phi3:mini

# Modelo de embeddings para búsqueda vectorial
ollama pull nomic-embed-text
```

**Nota:** Asegúrate de que Ollama esté ejecutándose en segundo plano antes de usar el sistema. Puedes iniciarlo con `ollama serve` en una terminal separada.

### 5. Preparar la base de datos RAG (opcional)
Si es la primera vez o deseas actualizar la base de datos con nuevos documentos, ejecuta el script de precarga:

```bash
python RAG/preload_database.py
```

Este script procesará los documentos en `RAG/docs/` y `RAG/extras/`, generará embeddings y los almacenará en `RAG/chroma_db/`.

---

## Uso del Sistema

### Ejecutar la aplicación principal
Para usar el sistema completo (Clasificación CV + Consultas RAG), ejecuta:

```bash
python main.py
```

El programa te pedirá la ruta a una imagen de residuo. El sistema:
1. Clasificará la imagen usando el modelo de Visión por Computador.
2. Generará una consulta automática sobre cómo reciclar ese tipo de residuo.
3. Consultará la base de datos RAG para obtener una respuesta experta basada en documentación.

Ejemplo de uso:
```
Introduce la ruta de la imagen (o escribe 'salir'): path/to/image.jpg
[CV] Analizando imagen: path/to/image.jpg...
--- CNN Detectó: Plastico ---
[RAG] Respuesta del Asistente Experto:
Según la documentación, el plástico debe depositarse en el contenedor amarillo...
```

### Uso individual de componentes
- **Solo RAG**: Ejecuta scripts en la carpeta `RAG/` para consultas directas.
- **Solo CV**: Modifica `main.py` para usar solo la clasificación sin RAG.

---

## Estructura del Sistema

A continuación se describe la función de los archivos principales:

### Raíz del Proyecto
* **`main.py`**: Punto de entrada principal. Integra CV y RAG para clasificar imágenes y consultar reciclaje.
* **`environment.yml`**: Archivo de configuración del entorno Conda con todas las dependencias.

### Carpeta CV/
* **`cv_reciclaria.onnx`**: Modelo de Visión por Computador entrenado (exportado desde MATLAB).
* Otros archivos relacionados con el modelo.

### Carpeta RAG/
#### Configuración y Utilidades
* **`config.py`**: Archivo central de configuración. Define modelos de Ollama, rutas, etc.
* **`url.py`**: Lista de URLs para extraer información adicional.

#### Procesamiento de Datos (Ingestión)
* **`get_docs.py`**: Script para obtener documentos desde URLs.
* **`preproces_data.py`**: Funciones para analizar y procesar documentos.
* **`preload_database.py`**: **Script crítico**. Genera embeddings de documentos y los guarda en ChromaDB.

#### Ejecución y Consultas
* **`query.py`**: Lógica para búsqueda en ChromaDB y consultas al LLM via Ollama.
* **`main.py`**: Interfaz alternativa para consultas RAG directas.

#### Directorios
* **`chroma_db/`**: Base de datos vectorial persistente (creada automáticamente).
* **`docs/`**: Carpeta para documentos fuente (PDF, TXT, etc.).
* **`extras/`**: Documentos adicionales sobre reciclaje.
* **`evaluacion/`**: Datos para evaluación y posibles mejoras (no implementado).