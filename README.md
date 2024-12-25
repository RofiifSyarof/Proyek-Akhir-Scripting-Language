# Panduan Penggunaan Website Streamlit

### Cara Pertama (Menggunakan Link)
Langkah-langkah
1. Ketikkan link berikut ke browser yang Anda gunakan:
https://pytorch-klasifikasi-sampah.streamlit.app

**CATATAN!!!**

Link tidak aktif secara permanen, ada kemungkinan link akan dinonaktifkan secara otomatis oleh Streamlit apabila tidak ada aktivitas.

### Cara Kedua (Menggunakan Server Cloud Streamlit)
**CATATAN!!**
1. Membutuhkan akun GitHub.
2. Perlu mengunggah program ke akun Github.
3. Referensi video: https://www.youtube.com/watch?v=HKoOBiAaHGg

Langkah-langkah
1. Login ke akun Github.
2. Unggah program ke repositori Github.
3. Buka link https://streamlit.io/cloud.
4. Sign Up menggunakan akun Github yang memiliki repositori program.
5. Pilih "Create app" di pojok kanan atas layar.
6. Pilih "Deploy a public app from Github."
7. Di bagian "Repository," masukkan link repositori program.
8. Pastikan "Branch" yang digunakan sesuai dengan yang terdapat di repositori.
9. Pastikan "Main file path" mengakses file "app.py".
10. "App URL" adalah opsional, bisa diisi untuk memudahkan penamaan link.
11. Jika semua pengaturan sudah selesai, klik tombol "Deploy" di bawah.

### Cara Ketiga (Menjalankan Secara Lokal)
**CATATAN!!**
1. Perlu membuat dan menginstal environment Streamlit.
2. Perlu menginstal library python yang dibutuhkan.
3. Referensi cara instal Streamlit: https://docs.streamlit.io/get-started/installation

Langkah-langkah
1. Buat dan instal Streamlit.
2. Masukkan program ke folder instalasi Streamlit.
3. Instal library Python yang dibutuhkan/sesuai dengan file "requirements.txt"
4. jalankan file "app.py" menggunakan command Streamlit.