import requests
from bs4 import BeautifulSoup
import os
import re
import time
from url import URLS

# --- CONFIGURACIÓN ---
OUTPUT_DIR = "RAG/docs"  # Carpeta donde se guardarán los .txt


def clean_filename(title):
    """Convierte un título en un nombre de archivo válido."""
    return re.sub(r'[\\/*?:"<>|]', "", title).strip().replace(" ", "_")

def clean_text(text):
    """
    Limpia el texto para que sea digerible por el LLM.
    Elimina espacios extra, saltos de línea múltiples, etc.
    """
    # 1. Separamos el texto por líneas originales
    lines = text.split(". ")
    
    # 2. Limpiamos cada línea individualmente (quitamos espacios al inicio/final)
    # y filtramos las líneas vacías.
    cleaned_lines = []
    for line in lines:
        
        # Reemplazar múltiples espacios/tabs por un solo espacio
        text = re.sub(r'\s+', ' ', line)
        # Reemplazar espacios antes de puntos/comas
        text = re.sub(r'\s([?.!"])', r'\1',text)
        # Solo guardamos la línea si tiene contenido real
        if text:
            cleaned_lines.append(text)
            
    # 3. Unimos las líneas con un salto de línea para que sea un texto vertical
    # Usamos '\n' para separar frases o '\n\n' si prefieres párrafos muy marcados.
    return "\n".join(cleaned_lines)
    return text.strip()

def scrape_url(url):
    print(f"🔄 Descargando: {url}")
    
    # Headers para parecer un navegador real y evitar bloqueos (403 Forbidden)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Lanza error si la web falla
        
        # Parsear HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. ELIMINAR RUIDO (Scripts, Estilos, Menús, Pies de página)
        # Esto es CRÍTICO para que el LLM no lea código basura.
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose() # Elimina la etiqueta del árbol

        # 2. Extraer Título para el nombre del archivo
        page_title = soup.title.string if soup.title else "documento_sin_titulo"
        filename = clean_filename(page_title) + ".txt"

        # 3. Extraer Texto Principal (Priorizamos párrafos y encabezados)
        # Buscamos el contenedor principal si es posible (común en blogs/artículos)
        content_div = soup.find('main') or soup.find('article') or soup.body
        
        # Obtener texto separando bloques por saltos de línea
        raw_text = content_div.get_text(separator='\n\n')

        # 4. Limpieza final
        final_text = clean_text(raw_text)

        # Añadimos la URL al principio del texto para referencia del RAG
        final_content = f"FUENTE: {url}\nTITULO: {page_title}\n\n{final_text}"

        # 5. Guardar archivo
        file_path = os.path.join(OUTPUT_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_content)
        
        print(f"✅ Guardado en: {file_path}")
        return True

    except Exception as e:
        print(f"❌ Error descargando {url}: {e}")
        return False

def main():
    # Crear carpeta si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Carpeta '{OUTPUT_DIR}' creada.")

    success_count = 0
    for url in URLS:
        if scrape_url(url):
            success_count += 1
        # Pequeña pausa para no saturar servidores (cortesía web)
        time.sleep(1) 

    print(f"\n✨ Proceso terminado. {success_count}/{len(URLS)} documentos procesados.")

if __name__ == "__main__":
    main()