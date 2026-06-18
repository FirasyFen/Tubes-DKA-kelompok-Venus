"""
Sistem Fuzzy Mamdani & Sugeno - Prediksi Kualitas Wine (White Wine)
=====================================================================
Aplikasi Streamlit hasil gabungan dari dua notebook:
  - tubes_DKA_ver3.ipynb     -> mesin fuzzy (fuzzifikasi, rule base, defuzzifikasi
                                 Mamdani & Sugeno, visualisasi fungsi keanggotaan)
  - Menghitung_Akurasi.ipynb -> evaluasi akurasi Mamdani vs Sugeno pada dataset,
                                 confusion matrix, dan korelasi antar variabel

Variabel input yang dipakai (sesuai dataset winequality-white-indonesia.csv):
  free sulfur dioxide, residual sugar, citric acid, pH, sulphates, alcohol

Cara menjalankan:
    pip install streamlit numpy pandas matplotlib seaborn scikit-learn
    atau pip install -r requirements.txt
    streamlit run fuzzy_wine_quality_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

# =====================================================================
# 1. FUNGSI KEANGGOTAAN DASAR (segitiga & trapesium)
# =====================================================================


def perhitungan_triangle(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:
        return 0.0


def perhitungan_trapezoidal(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    elif x >= b and x <= c:
        return 1.0
    elif x >= a and x < b:
        return 1.0 if b == a else (x - a) / (b - a)
    else:
        return 1.0 if d == c else (d - x) / (d - c)


# =====================================================================
# 2. FUZZIFIKASI SETIAP VARIABEL INPUT
# =====================================================================


def nilai_fsd(x):
    """Free Sulfur Dioxide"""
    return {
        "tinggi": perhitungan_trapezoidal(x, 45, 60, 100, 100),
        "sedang": perhitungan_trapezoidal(x, 20, 30, 45, 60),
        "rendah": perhitungan_trapezoidal(x, 0, 0, 20, 30),
    }


def nilai_rs(x):
    """Residual Sugar"""
    return {
        "tinggi": perhitungan_trapezoidal(x, 15, 20, 70, 70),
        "sedang": perhitungan_trapezoidal(x, 4, 5, 15, 20),
        "rendah": perhitungan_trapezoidal(x, 0, 0, 4, 5),
    }


def nilai_ca(x):
    """Citric Acid"""
    return {
        "tinggi": perhitungan_trapezoidal(x, 0.40, 0.50, 1.66, 1.66),
        "sedang": perhitungan_trapezoidal(x, 0.20, 0.30, 0.40, 0.50),
        "rendah": perhitungan_trapezoidal(x, 0.00, 0.00, 0.20, 0.30),
    }


def nilai_ph(x):
    """pH"""
    return {
        "basa": perhitungan_trapezoidal(x, 3.4, 3.5, 4, 4),
        "normal": perhitungan_trapezoidal(x, 3, 3.1, 3.4, 3.5),
        "asam": perhitungan_trapezoidal(x, 2.7, 2.7, 3, 3.1),
    }


def nilai_sul(x):
    """Sulphates"""
    return {
        "tinggi": perhitungan_trapezoidal(x, 0.7, 0.8, 1.2, 1.2),
        "sedang": perhitungan_trapezoidal(x, 0.4, 0.5, 0.7, 0.8),
        "rendah": perhitungan_trapezoidal(x, 0.2, 0.2, 0.4, 0.5),
    }


def nilai_al(x):
    """Alcohol"""
    return {
        "tinggi": perhitungan_trapezoidal(x, 11, 12, 15, 15),
        "sedang": perhitungan_triangle(x, 9, 10.5, 12),
        "rendah": perhitungan_trapezoidal(x, 8, 8, 9, 10),
    }


# =====================================================================
# 3. FUZZIFIKASI OUTPUT (QUALITY) & KATEGORI CRISP
# =====================================================================

RANGE_QUALITAS = np.arange(1, 10.01, 0.1)


def qualitas(x):
    return {
        "buruk": perhitungan_trapezoidal(x, 1, 1, 2.5, 4),
        "standar": perhitungan_triangle(x, 4, 5.5, 7),
        "bagus": perhitungan_trapezoidal(x, 7, 8.5, 10, 10),
    }


def kategori_dari_nilai(v):
    """Kategori hasil defuzzifikasi (Mamdani/Sugeno)."""
    if v < 4:
        return "buruk"
    elif v <= 7:
        return "standar"
    else:
        return "bagus"


def kategori_quality_asli(v):
    """Kategori label kualitas asli pada dataset."""
    if v <= 4:
        return "buruk"
    elif v <= 7:
        return "standar"
    else:
        return "bagus"


# =====================================================================
# 4. MESIN INFERENSI FUZZY (RULE BASE + DEFUZZIFIKASI)
# =====================================================================


def proses_fuzzy(fsd, rs, ca, p, s, a):
    free_sulfur_dioxide = nilai_fsd(fsd)
    residual_sugar = nilai_rs(rs)
    citric_acid = nilai_ca(ca)
    ph = nilai_ph(p)
    sulphates = nilai_sul(s)
    alkohol = nilai_al(a)

    # ---- Aturan kualitas BURUK ----
    r1 = min(alkohol["rendah"], sulphates["rendah"])
    r2 = min(free_sulfur_dioxide["tinggi"], alkohol["rendah"])
    r3 = min(ph["asam"], alkohol["rendah"])
    r4 = min(citric_acid["rendah"], sulphates["rendah"])
    r5 = min(alkohol["rendah"], citric_acid["rendah"], free_sulfur_dioxide["tinggi"])

    # ---- Aturan kualitas STANDAR ----
    r6 = min(ph["normal"], residual_sugar["sedang"], alkohol["sedang"])
    r7 = min(alkohol["sedang"], free_sulfur_dioxide["sedang"])
    r8 = min(ph["normal"], sulphates["sedang"])
    r9 = min(residual_sugar["sedang"], alkohol["sedang"])
    r10 = citric_acid["sedang"]

    # ---- Aturan kualitas BAGUS ----
    r11 = min(alkohol["tinggi"], sulphates["tinggi"])
    r12 = min(alkohol["tinggi"], free_sulfur_dioxide["rendah"])
    r13 = min(alkohol["tinggi"], ph["normal"])
    r14 = min(citric_acid["tinggi"], sulphates["tinggi"])
    r15 = min(alkohol["tinggi"], citric_acid["tinggi"], free_sulfur_dioxide["rendah"])

    alpha_buruk = max(r1, r2, r3, r4, r5)
    alpha_standar = max(r6, r7, r8, r9, r10)
    alpha_bagus = max(r11, r12, r13, r14, r15)

    # ---- Defuzzifikasi Mamdani (centroid) ----
    clip_buruk, clip_standar, clip_bagus = [], [], []
    tot_Bi, tot_zxBi = 0.0, 0.0
    for nilai in RANGE_QUALITAS:
        mu = qualitas(nilai)
        cb = min(alpha_buruk, mu["buruk"])
        cs = min(alpha_standar, mu["standar"])
        cg = min(alpha_bagus, mu["bagus"])
        clip_buruk.append(cb)
        clip_standar.append(cs)
        clip_bagus.append(cg)
        z = max(cb, cs, cg)
        tot_Bi += z
        tot_zxBi += nilai * z

    mamdani = tot_zxBi / tot_Bi if tot_Bi != 0 else 0.0

    # ---- Defuzzifikasi Sugeno (weighted average, nilai tetap = puncak MF) ----
    tot_Bi_s = alpha_bagus + alpha_buruk + alpha_standar
    sugeno = (
        ((alpha_bagus * 8.5) + (alpha_standar * 5.5) + (alpha_buruk * 2.5)) / tot_Bi_s
        if tot_Bi_s != 0
        else 0.0
    )

    return {
        "fuzzifikasi": {
            "free_sulfur_dioxide": free_sulfur_dioxide,
            "residual_sugar": residual_sugar,
            "citric_acid": citric_acid,
            "ph": ph,
            "sulphates": sulphates,
            "alkohol": alkohol,
        },
        "rules": [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15],
        "alpha_buruk": alpha_buruk,
        "alpha_standar": alpha_standar,
        "alpha_bagus": alpha_bagus,
        "mamdani": mamdani,
        "sugeno": sugeno,
        "kategori_mamdani": kategori_dari_nilai(mamdani),
        "kategori_sugeno": kategori_dari_nilai(sugeno),
        "clip_buruk": clip_buruk,
        "clip_standar": clip_standar,
        "clip_bagus": clip_bagus,
    }


# =====================================================================
# 5. EVALUASI DATASET (akurasi, confusion matrix)
# =====================================================================

KOLOM_DIBUTUHKAN = [
    "free sulfur dioxide",
    "residual sugar",
    "citric acid",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def evaluasi_dataset(df: pd.DataFrame):
    records = []
    for idx in range(len(df)):
        fsd = float(df["free sulfur dioxide"][idx])
        rs = float(df["residual sugar"][idx])
        ca = float(df["citric acid"][idx])
        p = float(df["pH"][idx])
        s = float(df["sulphates"][idx])
        a = float(df["alcohol"][idx])
        q = float(df["quality"][idx])

        hasil = proses_fuzzy(fsd, rs, ca, p, s, a)

        records.append(
            {
                "idx": idx,
                "kualitas_asli": q,
                "kualitas_mamdani": round(hasil["mamdani"], 4),
                "kualitas_sugeno": round(hasil["sugeno"], 4),
                "offset_mamdani": round(abs(q - hasil["mamdani"]), 4),
                "offset_sugeno": round(abs(q - hasil["sugeno"]), 4),
                "alpha_buruk": round(hasil["alpha_buruk"], 4),
                "alpha_standar": round(hasil["alpha_standar"], 4),
                "alpha_bagus": round(hasil["alpha_bagus"], 4),
                "kategori_mamdani": hasil["kategori_mamdani"],
                "kategori_sugeno": hasil["kategori_sugeno"],
            }
        )

    hasil_df = pd.DataFrame(records)
    hasil_df["kategori_asli"] = hasil_df["kualitas_asli"].apply(kategori_quality_asli)

    akurasi_mamdani = (hasil_df["kategori_mamdani"] == hasil_df["kategori_asli"]).mean() * 100
    akurasi_sugeno = (hasil_df["kategori_sugeno"] == hasil_df["kategori_asli"]).mean() * 100

    akurasi_mamdani_kualitas = (
        abs(hasil_df["kualitas_mamdani"] - hasil_df["kualitas_asli"]) <= 1
    ).mean() * 100
    akurasi_sugeno_kualitas = (
        abs(hasil_df["kualitas_sugeno"] - hasil_df["kualitas_asli"]) <= 1
    ).mean() * 100

    metrik = {
        "akurasi_mamdani_kategori": akurasi_mamdani,
        "akurasi_sugeno_kategori": akurasi_sugeno,
        "akurasi_mamdani_kualitas": akurasi_mamdani_kualitas,
        "akurasi_sugeno_kualitas": akurasi_sugeno_kualitas,
    }
    return hasil_df, metrik


# =====================================================================
# 6. FUNGSI VISUALISASI
# =====================================================================

# Versi vektor (untuk menggambar kurva fungsi keanggotaan)
triangle_vec = np.vectorize(perhitungan_triangle)
trapez_vec = np.vectorize(perhitungan_trapezoidal)


def plot_membership(x_vals, mf_dict, judul, xlabel, nilai_input=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    warna = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
    for i, (label, y) in enumerate(mf_dict.items()):
        ax.plot(x_vals, y, label=label.capitalize(), color=warna[i % len(warna)], linewidth=2)
    if nilai_input is not None:
        ax.axvline(nilai_input, color="black", linestyle=":", linewidth=1.5,
                    label=f"Input = {nilai_input:.2f}")
    ax.set_title(f"Fungsi Keanggotaan: {judul}", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("\u03bc(x)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_defuzzifikasi(hasil):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = RANGE_QUALITAS
    mf_buruk = trapez_vec(x, 1, 1, 2.5, 4)
    mf_standar = triangle_vec(x, 4, 5.5, 7)
    mf_bagus = trapez_vec(x, 7, 8.5, 10, 10)

    ax.plot(x, mf_buruk, "r--", linewidth=1.2, alpha=0.5, label="MF Buruk (asli)")
    ax.plot(x, mf_standar, "--", color="orange", linewidth=1.2, alpha=0.5, label="MF Standar (asli)")
    ax.plot(x, mf_bagus, "g--", linewidth=1.2, alpha=0.5, label="MF Bagus (asli)")

    ax.fill_between(x, hasil["clip_buruk"], alpha=0.35, color="red",
                     label=f'Clip Buruk (\u03b1={hasil["alpha_buruk"]:.3f})')
    ax.fill_between(x, hasil["clip_standar"], alpha=0.35, color="orange",
                     label=f'Clip Standar (\u03b1={hasil["alpha_standar"]:.3f})')
    ax.fill_between(x, hasil["clip_bagus"], alpha=0.35, color="green",
                     label=f'Clip Bagus (\u03b1={hasil["alpha_bagus"]:.3f})')

    ax.axvline(hasil["mamdani"], color="black", linewidth=2,
               label=f'Defuzz Mamdani = {hasil["mamdani"]:.3f}')
    ax.axvline(hasil["sugeno"], color="purple", linewidth=2, linestyle=":",
               label=f'Defuzz Sugeno = {hasil["sugeno"]:.3f}')

    ax.set_title("Visualisasi Clipping \u2014 Metode Centroid (Mamdani) & Sugeno", fontweight="bold")
    ax.set_xlabel("Skala Kualitas Wine")
    ax.set_ylabel("Derajat Keanggotaan")
    ax.set_xlim(1, 10)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# =====================================================================
# 7. STREAMLIT APP
# =====================================================================

st.set_page_config(page_title="Fuzzy Wine Quality (Mamdani & Sugeno)", layout="wide")

st.title("\U0001F377 Sistem Fuzzy Mamdani & Sugeno \u2014 Prediksi Kualitas White Wine")
st.caption(
    "Aplikasi hasil gabungan dari notebook `tubes_DKA_ver3.ipynb` (mesin fuzzy) "
    "dan `Menghitung_Akurasi.ipynb` (evaluasi akurasi pada dataset)."
)

tab_prediksi, tab_evaluasi = st.tabs(
    ["\U0001F52E Prediksi Tunggal", "\U0001F4CA Evaluasi Dataset (Akurasi)"]
)

# ---------------------------------------------------------------------
# TAB 1 : PREDIKSI TUNGGAL
# ---------------------------------------------------------------------
with tab_prediksi:
    st.subheader("Masukkan Parameter Wine")
    c1, c2, c3 = st.columns(3)
    with c1:
        fsd = st.number_input("Free Sulfur Dioxide (0 - 100)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
        rs = st.number_input("Residual Sugar (0 - 70)", min_value=0.0, max_value=70.0, value=6.0, step=0.1)
    with c2:
        ca = st.number_input("Citric Acid (0.0 - 1.66)", min_value=0.0, max_value=1.66, value=0.30, step=0.01)
        p = st.number_input("pH (2.7 - 4.0)", min_value=2.7, max_value=4.0, value=3.2, step=0.01)
    with c3:
        s = st.number_input("Sulphates (0.2 - 1.2)", min_value=0.2, max_value=1.2, value=0.5, step=0.01)
        a = st.number_input("Alcohol (%) (8.0 - 15.0)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)

    if st.button("Hitung Kualitas", type="primary"):
        st.session_state["hasil_prediksi"] = proses_fuzzy(fsd, rs, ca, p, s, a)
        st.session_state["input_prediksi"] = (fsd, rs, ca, p, s, a)

    if "hasil_prediksi" in st.session_state:
        hasil = st.session_state["hasil_prediksi"]
        fsd, rs, ca, p, s, a = st.session_state["input_prediksi"]

        st.markdown("### Hasil Inferensi (\u03b1-predikat)")
        m1, m2, m3 = st.columns(3)
        m1.metric("\u03b1 Buruk", f"{hasil['alpha_buruk']:.4f}")
        m2.metric("\u03b1 Standar", f"{hasil['alpha_standar']:.4f}")
        m3.metric("\u03b1 Bagus", f"{hasil['alpha_bagus']:.4f}")

        st.markdown("### Hasil Defuzzifikasi")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Mamdani (Centroid)", f"{hasil['mamdani']:.3f}")
            st.write(f"Kategori: **{hasil['kategori_mamdani']}**")
        with r2:
            st.metric("Sugeno (Weighted Average)", f"{hasil['sugeno']:.3f}")
            st.write(f"Kategori: **{hasil['kategori_sugeno']}**")

        st.markdown("### Visualisasi Defuzzifikasi")
        st.pyplot(plot_defuzzifikasi(hasil))

        with st.expander("Lihat Fungsi Keanggotaan Setiap Variabel Input"):
            x_fsd = np.linspace(0, 100, 300)
            x_rs = np.linspace(0, 30, 300)
            x_ca = np.linspace(0, 1.5, 300)
            x_ph = np.linspace(2.5, 4.2, 300)
            x_sul = np.linspace(0.1, 1.4, 300)
            x_al = np.linspace(7, 16, 300)

            g1, g2 = st.columns(2)
            with g1:
                st.pyplot(plot_membership(x_fsd, {
                    "rendah": trapez_vec(x_fsd, 0, 0, 20, 30),
                    "sedang": trapez_vec(x_fsd, 20, 30, 45, 60),
                    "tinggi": trapez_vec(x_fsd, 45, 60, 100, 100),
                }, "Free Sulfur Dioxide", "Nilai Free Sulfur Dioxide", fsd))

                st.pyplot(plot_membership(x_ca, {
                    "rendah": trapez_vec(x_ca, 0.00, 0.00, 0.20, 0.30),
                    "sedang": trapez_vec(x_ca, 0.20, 0.30, 0.40, 0.50),
                    "tinggi": trapez_vec(x_ca, 0.40, 0.50, 1.66, 1.66),
                }, "Citric Acid", "Nilai Citric Acid", ca))

                st.pyplot(plot_membership(x_sul, {
                    "rendah": trapez_vec(x_sul, 0.2, 0.2, 0.4, 0.5),
                    "sedang": trapez_vec(x_sul, 0.4, 0.5, 0.7, 0.8),
                    "tinggi": trapez_vec(x_sul, 0.7, 0.8, 1.2, 1.2),
                }, "Sulphates", "Nilai Sulphates", s))
            with g2:
                st.pyplot(plot_membership(x_rs, {
                    "rendah": trapez_vec(x_rs, 0, 0, 4, 5),
                    "sedang": trapez_vec(x_rs, 4, 5, 15, 20),
                    "tinggi": trapez_vec(x_rs, 15, 20, 70, 70),
                }, "Residual Sugar", "Nilai Residual Sugar", rs))

                st.pyplot(plot_membership(x_ph, {
                    "asam": trapez_vec(x_ph, 2.7, 2.7, 3.0, 3.1),
                    "normal": trapez_vec(x_ph, 3.0, 3.1, 3.4, 3.5),
                    "basa": trapez_vec(x_ph, 3.4, 3.5, 4.0, 4.0),
                }, "pH", "Nilai pH", p))

                st.pyplot(plot_membership(x_al, {
                    "rendah": trapez_vec(x_al, 8, 8, 9, 10),
                    "sedang": triangle_vec(x_al, 9, 10.5, 12),
                    "tinggi": trapez_vec(x_al, 11, 12, 15, 15),
                }, "Alcohol", "Kadar Alkohol (%)", a))
    else:
        st.info("Atur parameter di atas lalu klik **Hitung Kualitas**.")

# ---------------------------------------------------------------------
# TAB 2 : EVALUASI DATASET / AKURASI
# ---------------------------------------------------------------------
with tab_evaluasi:
    st.subheader("Evaluasi Akurasi pada Dataset")
    st.write(
        "Unggah file CSV dengan kolom: `free sulfur dioxide`, `residual sugar`, "
        "`citric acid`, `pH`, `sulphates`, `alcohol`, `quality` "
        "(format seperti `winequality-white-indonesia.csv`)."
    )

    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])
    col_a, col_b = st.columns(2)
    with col_a:
        sep = st.selectbox("Separator", [";", ","], index=0)
    with col_b:
        decimal = st.selectbox("Tanda desimal", [",", "."], index=0)

    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=sep, decimal=decimal)
        st.write(f"Jumlah baris: **{len(df)}**")
        st.dataframe(df.head())

        kolom_kurang = set(KOLOM_DIBUTUHKAN) - set(df.columns)
        if kolom_kurang:
            st.error(f"Kolom berikut belum ada di dataset: {kolom_kurang}")
        else:
            if st.button("Jalankan Evaluasi", type="primary"):
                with st.spinner("Menghitung fuzzy inference untuk setiap baris..."):
                    hasil_df, metrik = evaluasi_dataset(df)
                st.session_state["hasil_df"] = hasil_df
                st.session_state["metrik"] = metrik
                st.session_state["df_evaluasi"] = df

            if "hasil_df" in st.session_state:
                hasil_df = st.session_state["hasil_df"]
                metrik = st.session_state["metrik"]
                df_eval = st.session_state["df_evaluasi"]

                st.markdown("### Hasil Akurasi")
                ak1, ak2 = st.columns(2)
                with ak1:
                    st.metric("Akurasi Mamdani (kategori)", f"{metrik['akurasi_mamdani_kategori']:.2f}%")
                    st.metric("Akurasi Mamdani (kualitas \u00b11)", f"{metrik['akurasi_mamdani_kualitas']:.2f}%")
                with ak2:
                    st.metric("Akurasi Sugeno (kategori)", f"{metrik['akurasi_sugeno_kategori']:.2f}%")
                    st.metric("Akurasi Sugeno (kualitas \u00b11)", f"{metrik['akurasi_sugeno_kualitas']:.2f}%")

                st.markdown("### Confusion Matrix")
                labels = ["buruk", "standar", "bagus"]
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.write("**Mamdani**")
                    cm_mamdani = confusion_matrix(hasil_df["kategori_asli"], hasil_df["kategori_mamdani"], labels=labels)
                    cm_mamdani_df = pd.DataFrame(cm_mamdani,
                                                  index=[f"Asli {i}" for i in labels],
                                                  columns=[f"Pred {i}" for i in labels])
                    st.dataframe(cm_mamdani_df)
                with cm2:
                    st.write("**Sugeno**")
                    cm_sugeno = confusion_matrix(hasil_df["kategori_asli"], hasil_df["kategori_sugeno"], labels=labels)
                    cm_sugeno_df = pd.DataFrame(cm_sugeno,
                                                 index=[f"Asli {i}" for i in labels],
                                                 columns=[f"Pred {i}" for i in labels])
                    st.dataframe(cm_sugeno_df)

                st.markdown("### Detail Hasil per Baris")
                st.dataframe(hasil_df, use_container_width=True)

                csv_bytes = hasil_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Hasil (CSV)", data=csv_bytes,
                    file_name="hasil_fuzzy_evaluasi.csv", mime="text/csv",
                )

                st.markdown("### Distribusi Kualitas Asli")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(hasil_df["kualitas_asli"], bins=10, color="tab:blue", edgecolor="black", alpha=0.8)
                ax.set_title("Distribusi Data Kualitas Wine (Asli)")
                ax.set_xlabel("Quality")
                ax.set_ylabel("Frekuensi")
                st.pyplot(fig)
    else:
        st.info("Silakan upload file dataset CSV untuk menjalankan evaluasi akurasi.")
