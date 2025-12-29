# main.py

import argparse
from query import query_stream
import time

def help():
    print('''
--- Ayuda del Sistema RAG ---
Este sistema permite hacer preguntas sobre documentos PDF previamente procesados y almacenados en una base de datos vectorial.
Instrucciones:
    1. Asegúrate de tener la base de datos vectorial generada. Si no la tienes, ejecuta preload_database.py.
    2. Ejecuta este script principal: python main.py
    3. Escribe tus preguntas cuando se te solicite.
    4. Para salir del sistema, escribe "exit".
''')
def main():
    question = ""
    while question != "exit":
        if not question:
            print("\n--- Sistema RAG ---")
        question = input("Escribe tu pregunta: ")
        if question == "exit":
            print("Saliendo...")
            break
        if question == "help":
            help()
            continue
        start=time.time()   
        query_stream(question)
        end=time.time()
        print(f"(Tiempo de respuesta: {end-start:.2f} segundos)")


if __name__ == "__main__":
    main()