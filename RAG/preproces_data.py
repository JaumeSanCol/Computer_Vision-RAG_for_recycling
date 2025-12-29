import fitz 
from langchain_core.documents import Document

def analizar_fuentes(pdf_path, pagina_num):
    doc = fitz.open(pdf_path)
    pagina = doc[pagina_num - 1] # Las páginas empiezan en 0
    
    # Extraemos bloques de texto con información detallada
    dict_pagina = pagina.get_text("dict")
    
    for bloque in dict_pagina["blocks"]:
        if "lines" in bloque:
            for linea in bloque["lines"]:
                for span in linea["spans"]:
                    # Imprimimos el tamaño y el texto para identificar patrones
                    print(f"Tamaño: {span['size']:.2f} | Texto: {span['text'][:50]}")




def extraer_por_titulos(pdf_path, tamano_titulo=21.96):
    doc = fitz.open(pdf_path)
    fragmentos = []
    contenido_actual = ""
    metadata_actual = {}

    for pagina in doc:
        blocks = pagina.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: continue
            for linea in b["lines"]:
                for span in linea["spans"]:
                    # Si el tamaño es el de un título principal
                    if round(span["size"], 2) == tamano_titulo:
                        # Guardamos lo anterior si existe
                        if contenido_actual:
                            fragmentos.append(Document(page_content=contenido_actual, metadata=metadata_actual))
                        
                        # Iniciamos nuevo bloque
                        contenido_actual = span["text"] + "\n"
                        metadata_actual = {"titulo": span["text"], "fuente": pdf_path}
                    else:
                        # Añadimos texto al bloque actual
                        contenido_actual += span["text"] + " "
    
    # Añadir el último bloque
    if contenido_actual:
        fragmentos.append(Document(page_content=contenido_actual, metadata=metadata_actual))
    return fragmentos

if __name__ == "__main__":
    # Prueba la guia
    analizar_fuentes("RAG/docs/guia.pdf", 7) #->> tamaño de titulo que nos intersa es 21.96