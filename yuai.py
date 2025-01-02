import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Muat model dan tokenizer
model_path = "C:\\Users\\Getzbie Alfredo Tpoy\\Documents\\7_SMT_UNUD\\Projek Summarization\\model_summarization"  # Ganti dengan path model Anda
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# CSS untuk mengubah tampilan Streamlit
page_bg_color = """
<style>
.stApp {
    background-color: #FBFBFB; /* Ganti dengan kode warna yang Anda inginkan */
}
</style>
"""

# Terapkan CSS ke Streamlit
st.markdown(page_bg_color, unsafe_allow_html=True)

# Fungsi untuk melakukan summarization
def summarize_text(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Fungsi untuk membaca file dan mengembalikan teksnya
def read_file(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.read().decode('utf-8')
    return None

# Fungsi untuk menyimpan hasil summarization ke dalam file
def save_summary_to_file(summary, filename="summary.txt"):
    with open(filename, "w") as f:
        f.write(summary)
    return filename

# Judul aplikasi
st.title("Text Summarization")

# Pilihan input: Teks atau File
input_option = st.radio("Pilih tipe input:", ("Teks", "File"))

# Input teks atau file
input_text = ""
if input_option == "Teks":
    input_text = st.text_area("Masukkan teks yang akan diringkas:")
else:
    uploaded_file = st.file_uploader("Upload file teks:", type=["txt"])
    if uploaded_file is not None:
        input_text = read_file(uploaded_file)

# Jika ada input teks (baik dari teks area atau file), tampilkan tombol untuk summarization
if input_text:
    if st.button("Ringkas Teks"):
        summary = summarize_text(input_text)

        # Menampilkan output: Teks atau Unduh File
        output_option = st.radio("Pilih tipe output:", ("Tampilkan Teks", "Unduh File"))

        if output_option == "Tampilkan Teks":
            st.subheader("Hasil Ringkasan:")
            st.write(summary)
        else:
            # Simpan ke file dan berikan tautan unduh
            filename = save_summary_to_file(summary)
            with open(filename, "rb") as f:
                st.download_button(
                    label="Unduh Ringkasan",
                    data=f,
                    file_name="summary.txt",
                    mime="text/plain"
                )

# Berikan peringatan jika tidak ada input
else:
    st.warning("Silakan masukkan teks atau unggah file untuk diringkas.")
