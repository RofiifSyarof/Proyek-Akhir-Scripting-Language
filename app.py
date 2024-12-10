import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Definisi arsitektur model
class WasteClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(WasteClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    model = WasteClassifier(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Fungsi untuk memproses gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Tambahkan dimensi batch

# Fungsi untuk membuat prediksi
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        return predicted_class.item(), confidence.item(), probabilities

# Daftar nama kelas
CLASS_NAMES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

# Title aplikasi
st.title("Aplikasi Klasifikasi Sampah")

# Deskripsi aplikasi
st.write("""
Upload gambar sampah, dan aplikasi akan memprediksi jenis sampah beserta tingkat kepercayaan.
""")

# Memuat model
model_path = "model/best_cnn_model.pth"  # Ubah path ini sesuai file model Anda
model = load_model(model_path)

# Input gambar dari pengguna
uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    
    # Memproses gambar
    image_tensor = preprocess_image(image)
    
    # Prediksi
    predicted_class, confidence, probabilities = predict(model, image_tensor, CLASS_NAMES)
    
    # Menampilkan hasil prediksi
    st.subheader(f"**Prediksi:** {CLASS_NAMES[predicted_class]}")
    st.subheader(f"**Tingkat Kepercayaan:** {confidence * 100:.2f}%")
    
    # Menampilkan probabilitas untuk semua kelas
    st.write("**Probabilitas untuk setiap kelas:**")
    for idx, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {probabilities[idx].item() * 100:.2f}%")