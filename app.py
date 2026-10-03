import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components

st.set_page_config(
    page_title="CNN Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #080b12;
        color: white;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
        color: white;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #aab2c0;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 18px;
        background: rgba(25, 30, 42, 0.85);
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-digit {
        font-size: 90px;
        font-weight: 800;
        line-height: 1;
        color: white;
    }

    .prediction-label {
        font-size: 20px;
        color: #aab2c0;
        margin-top: 10px;
    }

    .confidence {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 10px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(20, 24, 34, 0.85);
        border-radius: 15px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

three_js = """
<!DOCTYPE html>
<html>
<head>
<style>
html, body {
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: transparent;
}

canvas {
    display: block;
}
</style>
</head>
<body>
<script type="module">
import * as THREE from
"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
});

renderer.setSize(
    window.innerWidth,
    window.innerHeight
);

renderer.setPixelRatio(
    Math.min(window.devicePixelRatio, 2)
);

document.body.appendChild(renderer.domElement);

const geometry = new THREE.BufferGeometry();

const starCount = 1200;

const positions = new Float32Array(starCount * 3);

for (let i = 0; i < starCount * 3; i += 3) {
    positions[i] =
        (Math.random() - 0.5) * 25;

    positions[i + 1] =
        (Math.random() - 0.5) * 25;

    positions[i + 2] =
        (Math.random() - 0.5) * 20;
}

geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(
        positions,
        3
    )
);

const material = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.025,
    transparent: true,
    opacity: 0.8
});

const stars = new THREE.Points(
    geometry,
    material
);

scene.add(stars);

function animate() {
    requestAnimationFrame(animate);

    stars.rotation.y += 0.0004;
    stars.rotation.x += 0.0001;

    renderer.render(
        scene,
        camera
    );
}

animate();

window.addEventListener(
    "resize",
    () => {
        camera.aspect =
            window.innerWidth /
            window.innerHeight;

        camera.updateProjectionMatrix();

        renderer.setSize(
            window.innerWidth,
            window.innerHeight
        );
    }
);
</script>
</body>
</html>
"""

components.html(
    three_js,
    height=1,
    scrolling=False
)

st.markdown(
    '<div class="main-title">🔢 CNN Handwritten Digit Recognizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a handwritten digit and let the CNN recognize it'
    '</div>',
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent

possible_models = [
    BASE_DIR / "models" / "mnist_cnn.keras",
    BASE_DIR / "mnist_cnn.keras",
    BASE_DIR / "models" / "mnist_cnn.h5",
    BASE_DIR / "mnist_cnn.h5",
]

MODEL_PATH = None

for path in possible_models:
    if path.exists():
        MODEL_PATH = path
        break

@st.cache_resource
def load_cnn_model(model_path):
    return load_model(model_path)

if MODEL_PATH is None:
    st.error(
        "CNN model was not found. "
        "Please make sure mnist_cnn.keras exists inside the models folder."
    )
    st.stop()

try:
    model = load_cnn_model(str(MODEL_PATH))
except Exception as e:
    st.error(
        f"Unable to load the CNN model: {e}"
    )
    st.stop()

with st.expander("Model Information"):
    st.write("**Model:** CNN")
    st.write("**Dataset:** MNIST")
    st.write("**Input:** 28 × 28 grayscale image")
    st.write("**Classes:** 0 – 9")
    st.write(f"**Model file:** `{MODEL_PATH.name}`")

st.markdown(
    '<div class="section-title">📤 Upload Handwritten Digit</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="digit_uploader"
)

def preprocess_image(image):
    image = image.convert("L")

    image = ImageOps.autocontrast(image)

    arr = np.array(image)

    border_pixels = np.concatenate(
        [
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ]
    )

    border_mean = np.mean(border_pixels)

    if border_mean > 127:
        arr = 255 - arr

    mask = arr > 25

    if not np.any(mask):
        return None, None

    ys, xs = np.where(mask)

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    digit = arr[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    h, w = digit.shape

    padding = int(
        max(h, w) * 0.20
    )

    canvas_size = max(h, w) + 2 * padding

    canvas = np.zeros(
        (canvas_size, canvas_size),
        dtype=np.uint8
    )

    y_offset = (
        canvas_size - h
    ) // 2

    x_offset = (
        canvas_size - w
    ) // 2

    canvas[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = digit

    digit_image = Image.fromarray(canvas)

    digit_image = digit_image.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    final_image.paste(
        digit_image,
        (4, 4)
    )

    final_image = final_image.filter(
        ImageFilter.GaussianBlur(radius=0.3)
    )

    processed = np.array(
        final_image
    ).astype("float32") / 255.0

    processed = processed.reshape(
        1,
        28,
        28,
        1
    )

    return final_image, processed

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        st.markdown(
            '<div class="section-title">'
            '🖼️ Uploaded Image'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original uploaded image",
                width=350
            )

        processed_image, processed_array = preprocess_image(image)

        if processed_image is None:
            st.error(
                "No handwritten digit was detected in the image."
            )
            st.stop()

        with col2:
            st.image(
                processed_image,
                caption="Processed MNIST-style image",
                width=350
            )

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🤖 CNN Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        prediction = model.predict(
            processed_array,
            verbose=0
        )

        predicted_digit = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            np.max(prediction[0])
        ) * 100

        st.markdown(
            f"""
<div class="prediction-box">
    <div class="prediction-digit">{predicted_digit}</div>
    <div class="prediction-label">Predicted Digit</div>
    <div class="confidence">Confidence: {confidence:.2f}%</div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '📊 Class Probabilities'
            '</div>',
            unsafe_allow_html=True
        )

        probabilities = prediction[0]

        for digit in range(10):

            probability = float(
                probabilities[digit]
            )

            col_a, col_b, col_c = st.columns(
                [1, 5, 1]
            )

            with col_a:
                st.write(
                    f"**{digit}**"
                )

            with col_b:
                st.progress(
                    probability
                )

            with col_c:
                st.write(
                    f"{probability * 100:.2f}%"
                )

        st.markdown("---")

        with st.expander(
            "View Raw Prediction Values"
        ):
            st.write(
                probabilities
            )

    except Exception as e:
        st.error(
            f"Error processing image: {e}"
        )

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#777;
        padding:15px;
    ">
        CNN Handwritten Digit Recognition • MNIST
    </div>
    """,
    unsafe_allow_html=True
)