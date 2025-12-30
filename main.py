import sys
import os
import onnxruntime as ort #type: ignore
import numpy as np #type: ignore
from PIL import Image


# Añadimos la carpeta RAG al sistema para usar query.py
directorio_actual = os.getcwd()
sys.path.append(os.path.join(directorio_actual, 'RAG'))

# Importamos la cadena RAG del archivo RAG/query.py
try:
    from query import rag_chain
except ImportError as e:
    print(f"Error al importar el RAG: {e}")
    sys.exit(1)

# Importamos el modelo ONNX exportado desde MATLAB
model_path = os.path.join("CV", "cv_reciclaria.onnx")

if not os.path.exists(model_path):
    print(f"Error: No se encuentra el modelo en {model_path}")
    sys.exit(1)

session = ort.InferenceSession(model_path)

 # Definimos las clases según el modelo entrenado en MATLAB
clases = {
    0: "Carton",
    1: "Organico", 
    2: "Vidrio",
    3: "Metal",
    4: "Basura",
    5: "Papel",
    6: "Plastico",
    7: "Textil",
    8: "Vegetacion"
}

def preprocesar_imagen(ruta_img):
    """Ajusta la imagen al formato que espera tu red de MATLAB."""
    # Abrir imagen, convertir a RGB y redimensionar a 224x224
    img = Image.open(ruta_img).convert('RGB')
    img = img.resize((224, 224)) 
    
    # Normalizar y convertir a float32
    img_data = np.array(img).astype(np.float32)
    
    # Cambiar formato de [H, W, C] a [C, H, W] que es lo que suele exportar MATLAB
    img_data = np.transpose(img_data, (2, 0, 1)) 
    return np.expand_dims(img_data, axis=0)

def clasificar_y_consultar(ruta_img):
    if not os.path.exists(ruta_img):
        print(f"Error: La imagen '{ruta_img}' no existe.")
        return

    # A. Inferencia de Visión por Computador
    print(f"\n[CV] Analizando imagen: {ruta_img}...")
    input_data = preprocesar_imagen(ruta_img)
    inputs = {session.get_inputs()[0].name: input_data}
    output = session.run(None, inputs)
    
    idx_clase = np.argmax(output[0])
    objeto_detectado = clases.get(idx_clase, "objeto desconocido")
    
    print(f"--- CNN Detectó: {objeto_detectado} ---")
    
    # B. Integración con el sistema RAG
    pregunta_automatica = f"¿Cómo debo reciclar correctamente un residuo de {objeto_detectado}?"
    
    print("\n[RAG] Respuesta del Asistente Experto:")
    # Usamos la 'rag_chain' definida en query.py
    for chunk in rag_chain.stream(pregunta_automatica):
        print(chunk, end="", flush=True)
    print("\n" + "-"*30)

if __name__ == "__main__":
    print("--- Bienvenido al Sistema ReciclarIA (CV + RAG) ---")
    while True:
        ruta = input("\nIntroduce la ruta de la imagen (o escribe 'salir'): ").strip()
        if ruta.lower() == 'salir':
            break
        clasificar_y_consultar(ruta)