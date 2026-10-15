import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path


# Nombre de las clases
CLASS_NAMES = [
    "Cartón",
    "Vidrio",
    "Metal",
    "Papel",
    "Plástico",
    "Basura"
]


# Configuración de la página
st.set_page_config(
    page_title="Clasificador de residuos",
    page_icon="♻️"
)


# Crear la CNN directamente. El archivo .keras disponible está corrupto y no
# debe abrirse con load_model(), porque eso provoca BadZipFile.
@st.cache_resource
def load_model():
    modelo_path = Path(__file__).parent / "modelo_residuos.keras"
    return tf.keras.models.load_model(modelo_path)


model = load_model()

# Título de la aplicación

st.title("♻️ Clasificador de residuos con CNN")


st.write(
    "Carga una imagen de un residuo y el modelo clasificará la categoría."
)



# Cargar imagen

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file is not None:


    # Leer imagen

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Imagen cargada",
        width=300
    )



    # Preparar imagen igual que en entrenamiento

    image_resize = image.resize(
        (224,224)
    )

    img_array = np.array(
        image_resize,
        dtype=np.float32
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )



    # Realizar predicción

    prediction = model.predict(
        img_array,
        verbose=0
    )


    prediction = prediction[0]


    predicted_class = np.argmax(
        prediction
    )


    confidence = np.max(
        prediction
    )



    # Mostrar resultado

    st.success(
        f"Clase predicha: {CLASS_NAMES[predicted_class]}"
    )


    st.info(
        f"Nivel de confianza: {confidence*100:.2f}%"
    )



    # Mostrar todas las probabilidades

    st.subheader(
        "Probabilidad por material"
    )


    for i in range(len(CLASS_NAMES)):

        st.write(
            f"{CLASS_NAMES[i]}: {prediction[i]*100:.2f}%"
        )
