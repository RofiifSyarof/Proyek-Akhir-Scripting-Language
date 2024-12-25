import streamlit as st
from streamlit_option_menu import option_menu
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

# Penjelasan untuk setiap klasifikasi sampah
CLASS_DESCRIPTIONS = {
    'baterai': """
    ### Informasi Umum

    Baterai bekas termasuk limbah berbahaya yang harus dikelola dengan hati-hati. Kandungan bahan kimia di dalamnya dapat mencemari tanah dan air jika dibuang sembarangan. 
    Anda bisa mengumpulkan baterai bekas untuk didaur ulang atau diserahkan ke fasilitas pengelolaan limbah elektronik. 
    
    ### Daur Ulang
    Banyak perusahaan elektronik atau bank sampah menerima baterai bekas untuk didaur ulang. Jangan lupa untuk memisahkan baterai dari limbah lain sebelum diserahkan!
    """,
    'organik': """
    ### Informasi Umum

    Sampah organik seperti sisa makanan dan daun kering dapat dengan mudah diubah menjadi kompos. Kompos ini bisa digunakan sebagai pupuk untuk tanaman. 
    Dengan mengolah sampah organik, Anda ikut mengurangi limbah yang dikirim ke tempat pembuangan akhir.

    ### Cara Membuat Kompos
    1. Kumpulkan sampah organik.
    2. Potong menjadi bagian kecil untuk mempercepat proses pengomposan.
    3. Masukkan ke dalam wadah kompos, tambahkan tanah, dan biarkan terurai secara alami.
    """,
    'kardus': """
    ### Informasi Umum

    Kardus adalah jenis sampah anorganik yang memiliki banyak manfaat. Kardus dapat didaur ulang menjadi produk baru atau dijadikan bahan kerajinan tangan yang menarik.
    
    ### Ide Kerajinan
    1. **Kastil**: Gunakan gulungan tisu bekas dan kotak kardus untuk membuat miniatur kastil.
    2. **Mainan Roket**: Buat mainan roket dari potongan kardus yang disusun.
    3. **Bingkai Foto**: Hiasi kardus menjadi bingkai foto unik.

    ### Daur Ulang
    Sampah kardus mudah dijual ke mitra pengumpul atau bank sampah terdekat.
    """,
    'kain': """
    ### Informasi Umum

    Sampah kain seperti pakaian bekas atau potongan kain dapat didaur ulang menjadi produk baru seperti tas, keset, atau bahkan baju baru.
    
    ### Ide Kreatif
    Gunakan kain bekas untuk membuat kerajinan tangan seperti patchwork atau hiasan dinding. Anda juga dapat menyumbangkan pakaian bekas yang masih layak pakai.
    """,
    'kaca': """
    ### Informasi Umum

    Sampah kaca dapat didaur ulang berkali-kali tanpa kehilangan kualitasnya. Contohnya adalah botol kaca, yang bisa digunakan kembali atau dilebur menjadi produk baru.
    
    ### Daur Ulang
    Pastikan kaca tidak pecah saat diserahkan ke fasilitas daur ulang. Jika pecah, gunakan wadah tertutup untuk menghindari bahaya.
    """,
    'metal': """
    ### Informasi Umum

    Sampah metal seperti kaleng aluminium atau besi tua memiliki nilai jual tinggi dan dapat didaur ulang menjadi produk baru.
    
    ### Daur Ulang
    Kumpulkan sampah metal dan pisahkan berdasarkan jenis logam. Serahkan ke bank sampah atau fasilitas daur ulang.
    """,
    'kertas': """
    ### Informasi Umum

    Kertas bekas adalah salah satu jenis limbah yang paling mudah didaur ulang. Kertas dapat diolah kembali menjadi kertas baru atau produk kreatif.
    
    ### Daur Ulang
    Potong kertas bekas menjadi kecil, rendam, dan buatlah bubur kertas untuk dijadikan kertas daur ulang.
    """,
    'plastik': """
    ### Informasi Umum

    Sampah plastik membutuhkan waktu lama untuk terurai. Namun, plastik dapat didaur ulang menjadi produk baru seperti pot bunga atau paving block.
    
    ### Daur Ulang
    Pilah plastik berdasarkan jenisnya dan serahkan ke fasilitas daur ulang atau bank sampah.
    """,
    'sepatu': """
    ### Informasi Umum

    Sepatu bekas yang masih layak pakai bisa disumbangkan. Jika tidak, sepatu dapat didaur ulang menjadi bahan baku produk baru.
    
    ### Daur Ulang
    Serahkan sepatu bekas ke komunitas atau program daur ulang khusus untuk sepatu.
    """,
    'lainnya': """
    ### Informasi Umum

    Sampah yang tidak termasuk kategori di atas bisa berupa bahan campuran. Pastikan untuk memisahkan limbah berbahaya dari limbah biasa.
    
    ### Saran
    Konsultasikan dengan bank sampah terdekat atau cari program daur ulang yang menerima limbah jenis ini.
    """
}


# Nama Halaman di Tab
st.set_page_config(page_title="Klasifikasi Jenis Sampah", page_icon="♻️", layout="centered")

# Menu horizontal
selected = option_menu(
    None, ['Home', 'Klasifikasi Sampah', 'Tentang'],
    icons=['house', 'recycle', 'info-circle'],
    menu_icon='cast', default_index=0, orientation='horizontal'
)

if selected == 'Home':
    st.markdown("""
        <h2 style="text-align: center;">♻️Aplikasi Klasifikasi Jenis Sampah♻️</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style="text-align: center;">
        Selamat datang di aplikasi Klasifikasi Sampah berbasis pembelajaran mesin! 
        Aplikasi ini membantu mengidentifikasi kategori sampah berdasarkan gambar yang Anda unggah.
        </p>
    """, unsafe_allow_html=True)

# Tambahkan logika di bagian Klasifikasi Sampah
elif selected == 'Klasifikasi Sampah':
    st.markdown("""
        <h2 style="text-align: center;">📸 Klasifikasi Sampah</h2>
    """, unsafe_allow_html=True)

    st.warning("""
         Aplikasi ini hanya menerima file gambar dengan format **JPG**, **JPEG**, dan **PNG**, 
        serta ukuran file maksimal **200 MB**. Pastikan file yang diunggah sesuai! ⚠️   
    """)

    model_path = "model/best_cnn_model.pth"
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            if uploaded_file.size > 200 * 1024 * 1024:  # 200 MB
                st.error("Ukuran file terlalu besar! Mohon unggah file dengan ukuran maksimal 200 MB.")
            else:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Gambar yang diupload", use_container_width=True)
                
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict(model, image_tensor, CLASS_NAMES)
                
                if confidence >= 0.6:
                    st.subheader(f"**Gambar yang Anda unggah kemungkinan merupakan jenis sampah** {CLASS_NAMES[predicted_class]}.")
                    st.subheader(f"**Kami memprediksinya dengan tingkat keyakinan sebesar** {confidence * 100:.2f}%")
                    
                    # Tampilkan deskripsi dari kategori sampah
                    category_description = CLASS_DESCRIPTIONS.get(CLASS_NAMES[predicted_class], "Informasi tidak tersedia.")
                    st.write(category_description)
                else:
                    st.subheader("Gambar yang Anda unggah kemungkinan bukan gambar sampah.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

elif selected == 'Tentang':
    st.markdown("""
        <h2 style="text-align: center;">Tentang Aplikasi</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h4>👨‍💻 Dikembangkan Oleh:</h4>
        <ul>
            <li>Muhammad Rofiif Syarof Nur Aufaa (22537141014)</li>
            <li>Wahyu Nur Cahyanto (22537141026)</li>
            <li>Muhammad Efflin Rizqallah Limbong (22537144007)</li>
        </ul>
        <p>Untuk memenuhi tugas akhir mata kuliah <b>Scripting Language</b>.</p>
    """, unsafe_allow_html=True)
