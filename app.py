import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

# Definisi kelas waste
WASTE_CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 
                 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model_path = "model/best_cnn_model.pth"  # Pastikan path sesuai dengan lokasi file model Anda
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Fungsi untuk memproses gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Sesuaikan dengan ukuran input model Anda
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Streamlit antarmuka pengguna
st.title("Waste Classification")
st.write("Upload an image of waste to classify its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing gambar
    input_tensor = preprocess_image(image)

    # Melakukan prediksi
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)

    # Menampilkan hasil prediksi
    predicted_label = WASTE_CLASSES[predicted_class.item()]
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence.item() * 100:.2f}%")

    # Menampilkan semua probabilitas
    st.subheader("Class Probabilities")
    for i, (cls, prob) in enumerate(zip(WASTE_CLASSES, probabilities)):
        st.write(f"{cls}: {prob.item() * 100:.2f}%")
