import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Muat model dan tokenizer
model_path = "./model_summarization"  # Ganti dengan path model Anda
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# CSS untuk mengubah tampilan Streamlit


# Terapkan CSS ke Streamlit


# Fungsi untuk melakukan summarization
def summarize_text(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"],
            min_length=20,
            max_length=80,
            num_beams=10,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True,
            do_sample = True,
            temperature = 0.8,
            top_k = 50,
            top_p = 0.95)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Judul aplikasi
st.title("Text Summarization")

# Input
input_text = st.text_area("Masukkan teks yang akan diringkas:", height=300)

# Jika ada input teks (baik dari teks area atau file), tampilkan tombol untuk summarization
if st.button("Ringkas Teks"):
        summary = summarize_text(input_text)

        st.subheader("Hasil Ringkasan:")
        st.write(summary)

# Berikan peringatan jika tidak ada input\
