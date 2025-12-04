**Laporan Singkat Implementasi Real-time rPPG**

**Mata Kuliah**: Sistem & Teknologi Multimedia

**Nama**: Hizkia Christovita Siahaan

**NIM**: 122140110

---
**1. Deskripsi Singkat**

Program ini mengimplementasikan remote Photoplethysmography (rPPG) secara real-time untuk mengukur detak jantung hanya menggunakan webcam. Sistem mengambil area dahi sebagai Region of Interest (ROI) dan mengekstraksi perubahan intensitas warna hijau yang terkait dengan aliran darah. Dengan menggunakan MediaPipe FaceMesh untuk tracking dahi yang stabil, program membangun sinyal rPPG secara kontinu, memprosesnya dengan filtering dan FFT, lalu menghasilkan estimasi BPM (Beats Per Minute) secara real-time. Implementasi ini juga dilengkapi motion detection, sinyal kualitas (SNR), smoothing BPM, serta statistik sesi pada akhir program.

---
**2.Pustaka Yang Digunakan**

Program memanfaatkan beberapa pustaka utama. OpenCV digunakan untuk menangkap frame webcam, menangani ROI, menampilkan indikator motion, SNR, BPM, serta visualisasi overlay. MediaPipe digunakan untuk mendeteksi wajah dan landmark wajah secara stabil, sehingga ROI dahi dapat mengikuti gerakan pengguna dengan akurat. NumPy digunakan untuk perhitungan numerik, seperti rata-rata kanal hijau dan operasi array sinyal. SciPy menyediakan bandpass filter Butterworth dan fungsi pemrosesan sinyal untuk membersihkan noise sebelum FFT. Matplotlib dipakai untuk menampilkan grafik sinyal real-time berupa Raw Signal, Filtered Signal, dan Power Spectrum. Pustaka standar seperti time dan collections.deque digunakan untuk buffering dan timestamping.

Dalam output terlihat bahwa TensorFlow Lite delegate dari MediaPipe juga dimuat otomatis, tetapi tidak digunakan secara eksplisit dalam kode—delegasi tersebut hanya bagian internal MediaPipe.

---
**3.Penjelasan Metode**

Pertama, webcam menangkap frame kemudian MediaPipe FaceMesh mendeteksi wajah dan mengambil titik landmark. Dari landmark tersebut, area dahi ditentukan sebagai ROI secara adaptif agar sinyal tetap stabil meski pengguna sedikit bergerak. Dari ROI diambil nilai rata-rata kanal hijau pada setiap frame, kemudian dimasukkan ke sliding window buffer selama beberapa detik. Sinyal mentah ini kemudian melalui tahap detrending dan bandpass filtering (0.67–4.0 Hz) untuk mengambil frekuensi detak jantung manusia. FFT digunakan untuk mencari frekuensi dominan dalam sinyal yang sudah difilter dan mengubahnya menjadi BPM menggunakan frekuensi × 60.

Untuk menjaga stabilitas estimasi, program menerapkan Exponential Moving Average, sehingga nilai BPM tidak fluktuatif meskipun intensitas sinyal berubah. Selain itu, program memantau tingkat pergerakan wajah; jika kondisi “Moving”, estimasi BPM ditunda untuk menjaga akurasi. Kualitas sinyal (Signal-to-Noise Ratio) juga dihitung sebagai acuan apakah estimasi BPM layak ditampilkan. Semua proses ini berlangsung secara real-time.

---
**4.Aspek Pembeda Dengan Demo Kelas**

Implementasi ini memiliki sejumlah fitur tambahan dibandingkan versi dasar yang dipelajari di kelas. Program menambahkan motion detector yang memutuskan apakah data sinyal layak diproses, sehingga mengurangi kesalahan estimasi saat wajah bergerak. Selain itu, terdapat indikator SNR untuk menilai kualitas sinyal. ROI dahi yang digunakan bersifat adaptif berdasarkan landmark, bukan bounding box statis seperti demo biasa. Hasil estimasi BPM juga disempurnakan menggunakan EMA smoothing, yang tidak ada pada versi dasar. Grafik streaming real-time (Raw, Filtered, FFT) juga memberi pengguna pemahaman visual terhadap proses sinyal rPPG. Terakhir, program menampilkan statistik sesi seperti total frame, rata-rata FPS, dan durasi—fitur yang tidak tersedia pada demo dasar.

---
**5.Credit AI**

LINK : https://chatgpt.com/share/6931ba58-b874-8009-9b89-a35663854d80
