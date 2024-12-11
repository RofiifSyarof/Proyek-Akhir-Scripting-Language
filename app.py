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
CLASS_NAMES = ['baterai', 'organik', 'kardus', 'kain', 'kaca',
               'metal', 'kertas', 'plastik', 'sepatu', 'lainnya']

# Nama Halaman di Tab
st.set_page_config(page_title="Klasifikasi Jenis Sampah", page_icon="♻️", layout="centered")

# Title aplikasi
st.markdown("""
    <h2 style="text-align: center;">♻️Aplikasi Klasifikasi Jenis Sampah♻️</h2>
""", unsafe_allow_html=True)


# Deskripsi aplikasi
st.html("""
    <h4>⚠️Bagaimana cara menggunakannya?</h4>
    <h5>1. Upload gambar yang ingin Anda deteksi.</h5>
    <h5>2. Tunggu situs untuk memproses gambar.</h5>
    <h5>3. Hasil prediksi akan ditampilkan di bawah gambar.</h5>
    <br>
    
    <h4>⚠️Catatan⚠️</h4>
    <h5>1. Aplikasi saat ini hanya menerima format gambar JPG. JPEG, dan PNG dengan ukuran maksimal 200MB.</h5>
""")

# Memuat model
model_path = "model/best_cnn_model.pth"  # Ubah path ini sesuai file model Anda
model = load_model(model_path)

# Input gambar dari pengguna
uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    # Memproses gambar
    image_tensor = preprocess_image(image)
    
    # Prediksi
    predicted_class, confidence, probabilities = predict(model, image_tensor, CLASS_NAMES)
    
    # Menampilkan hasil prediksi
    if confidence >= 0.6:
        st.subheader(f"**Gambar yang Anda unggah kemungkinan merupakan jenis sampah** {CLASS_NAMES[predicted_class]}.")
        st.subheader(f"**Kami memprediksinya dengan tingkat keyakinan sebesar** {confidence * 100:.2f}%")
    else:
        st.subheader("Gambar yang Anda unggah kemungkinan bukan gambar sampah.")
        
st.write('---')
st.markdown("""
            <h5>Dikembangkan oleh Kelompok 5, dengan anggota:
            <br><br>
            1.  Muhammad Rofiif Syarof Nur Aufaa (22537141014)<br>
            2.  Wahyu Nur Cahyanto (22537141026)<br>
            3.  Muhammad Efflin Rizqallah Limbong (22537144007)
            <br><br>
            Untuk memenuhi tugas akhir mata kuliah Scripting Language.
            </h5>
                """,
            unsafe_allow_html=True)