import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load static dataset
df = pd.read_csv('summarization.csv')

# Sidebar navigation
st.sidebar.title("News Option")
page = st.sidebar.radio("", ["Indonesian News Data", "Text Summarization"])

if page == "Indonesian News Data":
    st.title("Indonesian News Summarization")

    # Pilih topik
    selected_topic = st.selectbox("Pilih Topik", options=df["category"].unique())
    custom_topic = st.text_input("Ketik Topik")

    # Tentukan jumlah berita
    num_articles = st.slider("Jumlah berita", min_value=1, max_value=10, value=5)

    # Filter data
    if custom_topic:
        filtered_df = df[df["category"].str.contains(custom_topic, case=False, na=False)]
    else:
        filtered_df = df[df["category"] == selected_topic]

    filtered_df = filtered_df.head(num_articles)

    # Tampilkan sumber dan ringkasan
    st.subheader("Berita dan Ringkasan:")
    for _, row in filtered_df.iterrows():
        st.write(f"*Sumber:* [ {row['source']} ]({row['source_url']})")
        st.write(f"*Ringkasan:* {row['summary']}")
        st.write("---")

elif page == "Text Summarization":
    st.title("Text Summarization")

    # Load model dan tokenizer
    model_path = "./model_summarization"  # Ganti dengan path model Anda
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Input teks untuk diringkas
    input_text = st.text_area("Masukkan teks yang akan diringkas:", height=300)

    # Ringkasan teks
    if st.button("Ringkas Teks"):
        def summarize_text(text):
            inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(
                inputs["input_ids"],
                min_length=20,
                max_length=80,
                num_beams=10,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        summary = summarize_text(input_text)

        st.subheader("Hasil Ringkasan:")
        st.write(summary)