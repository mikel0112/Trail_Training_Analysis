import base64
import yaml
import os

# Función para convertir una imagen a base64
def imagen_a_base64(ruta_imagen):
    with open(ruta_imagen, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    

def outputs_images_list():
    outputs_folders = os.listdir("outputs")
    output_images = []
    for folder in outputs_folders:
        images = os.listdir(f"outputs/{folder}")
        for image in images:
            if image.endswith(".png") or image.endswith(".jpg"):
                output_images.append(f"outputs/{folder}/{image}")
    return output_images

if __name__ == "__main__":

    
    params = yaml.safe_load(open("params.yaml", "r"))
    nombre_atleta = params["personal_data"]["name"]
    html_path = f"outputs/Estadísticas{nombre_atleta}.html"
    if os.path.exists(html_path):
        os.remove(html_path)
    # Lista de rutas de las imágenes que quieres incluir
    imagenes = outputs_images_list()

    # Crear el HTML con botones para seleccionar cada imagen
    html_contenido = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Visualización de Datos Entrenamiento</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 20px;
            }
            img {
                max-width: 150%;
                height: auto;
                display: none;
                margin: 20px auto;
            }
            button {
                padding: 10px 20px;
                margin: 5px;
                font-size: 16px;
            }
        </style>
    </head>
    <body>

        <h1>Visualización de Datos Entrenamiento</h1>
    """
    #lista imágenes sin .png
    clean_lista_imagenes = []
    for imagen in imagenes:
        image_name = imagen.split("/")[-1]
        clean_lista_imagenes.append(image_name.replace(".png", ""))
    # Agregar botones y elementos img
    for i, ruta in enumerate(imagenes):
        imagen_base64 = imagen_a_base64(ruta)
        html_contenido += f"""
        <button onclick="mostrarImagen('imagen{i}')">{clean_lista_imagenes[i]}</button>
        <img id="imagen{i}" src="data:image/jpeg;base64,{imagen_base64}" alt="Imagen {i + 1}">
        """

    # Script JavaScript para mostrar la imagen seleccionada
    html_contenido += """
        <script>
            function mostrarImagen(id) {
                // Ocultar todas las imágenes
                document.querySelectorAll('img').forEach(img => img.style.display = 'none');
                // Mostrar la imagen seleccionada
                document.getElementById(id).style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

    # Guardar el HTML generado en un archivo
    with open(f"outputs/Estadísticas{nombre_atleta}.html", "w", encoding="utf-8") as file:
        file.write(html_contenido)

    print("Archivo HTML generado exitosamente como 'Estadísticas.html'")