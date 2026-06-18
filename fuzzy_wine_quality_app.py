"""
Sistem Fuzzy Mamdani & Sugeno - Prediksi Kualitas Wine
=======================================================
Aplikasi Streamlit hasil gabungan dari:
  - tubes_DKA_ver3.ipynb        -> engine fuzzy (fuzzifikasi, rule base, defuzzifikasi)
  - Menghitung_Akurasi.ipynb    -> evaluasi akurasi Mamdani vs Sugeno pada dataset

Cara menjalankan:
    pip install streamlit numpy pandas matplotlib
    streamlit run fuzzy_wine_quality_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# 1. FUNGSI KEANGGOTAAN DASAR
# =========================================================

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


triangle_vec = np.vectorize(perhitungan_triangle)
trapez_vec = np.vectorize(perhitungan_trapezoidal)

# =========================================================
# 2. FUZZIFIKASI SETIAP VARIABEL INPUT
# =========================================================

def nilai_fa(x):
    return {
        "tinggi": perhitungan_trapezoidal(x, 8, 10, 15, 15),
        "sedang": perhitungan_trapezoidal(x, 5, 6, 8, 9),
        "rendah": perhitungan_trapezoidal(x, 3, 3, 5, 6),
    }


def nilai_rs(x):
    return {
        "tinggi": perhitungan_trapezoidal(x, 15, 20, 70, 70),
        "sedang": perhitungan_trapezoidal(x, 4, 5, 15, 20),
        "rendah": perhitungan_trapezoidal(x, 0, 0, 4, 5),
    }


def nilai_den(x):
    return {
        "tinggi": perhitungan_trapezoidal(x, 0.998, 1, 1.04, 1.04),
        "normal": perhitungan_trapezoidal(x, 0.992, 0.994, 0.998, 1),
        "rendah": perhitungan_trapezoidal(x, 0.987, 0.987, 0.992, 0.994),
    }


def nilai_ph(x):
    return {
        "basa": perhitungan_trapezoidal(x, 3.4, 3.5, 4, 4),
        "normal": perhitungan_trapezoidal(x, 3, 3.1, 3.4, 3.5),
        "asam": perhitungan_trapezoidal(x, 2.7, 2.7, 3, 3.1),
    }


def nilai_sul(x):
    return {
        "tinggi": perhitungan_trapezoidal(x, 0.7, 0.8, 1.2, 1.2),
        "sedang": perhitungan_trapezoidal(x, 0.4, 0.5, 0.7, 0.8),
        "rendah": perhitungan_trapezoidal(x, 0.2, 0.2, 0.4, 0.5),
    }


def nilai_al(x):
    return {
        "tinggi": perhitungan_trapezoidal(x, 11, 12, 15, 15),
        "sedang": perhitungan_triangle(x, 9, 10.5, 12),
        "rendah": perhitungan_trapezoidal(x, 8, 8, 9, 10),
    }


# =========================================================
# 3. FUZZIFIKASI OUTPUT (QUALITY) & KATEGORI CRISP
# =========================================================

RANGE_QUALITAS = np.arange(1, 10.01, 0.1)


def qualitas(x):
    return {
        "buruk": perhitungan_trapezoidal(x, 1, 1, 2.5, 4),
        "standar": perhitungan_triangle(x, 4, 5.5, 7),
        "bagus": perhitungan_trapezoidal(x, 7, 8.5, 10, 10),
    }


def kategori_dari_nilai(v):
    if v >= 1 and v < 4:
        return "buruk"
    elif v >= 4 and v <= 7:
        return "standar"
    else:
        return "bagus"


def kategori_quality_asli(v):
    """Kategori label asli dataset (digunakan saat evaluasi akurasi)."""
    if v <= 4:
        return "buruk"
    elif v <= 7:
        return "standar"
    else:
        return "bagus"


# =========================================================
# 4. MESIN INFERENSI FUZZY (RULE BASE + DEFUZZIFIKASI)
# =========================================================

def proses_fuzzy(fa, rs, d, p, s, a):
    fixed_acidity = nilai_fa(fa)
    residual_sugar = nilai_rs(rs)
    densitas = nilai_den(d)
    ph = nilai_ph(p)
    sulphates = nilai_sul(s)
    alkohol = nilai_al(a)

    # --- Rule base: kualitas buruk ---
    r1 = min(alkohol["rendah"], sulphates["rendah"])
    r2 = min(densitas["tinggi"], alkohol["rendah"])
    r3 = min(ph["asam"], alkohol["rendah"])
    r4 = min(fixed_acidity["tinggi"], sulphates["rendah"])
    r5 = min(alkohol["rendah"], sulphates["rendah"], densitas["tinggi"])

    # --- Rule base: kualitas standar ---
    r6 = min(ph["normal"], residual_sugar["sedang"], alkohol["sedang"])
    r7 = min(alkohol["sedang"], densitas["normal"])
    r8 = min(ph["normal"], sulphates["sedang"])
    r9 = min(residual_sugar["sedang"], alkohol["sedang"])
    r10 = fixed_acidity["sedang"]

    # --- Rule base: kualitas bagus ---
    r11 = min(alkohol["tinggi"], sulphates["tinggi"])
    r12 = min(alkohol["tinggi"], densitas["rendah"])
    r13 = min(alkohol["tinggi"], ph["normal"])
    r14 = min(residual_sugar["tinggi"], sulphates["tinggi"])
    r15 = min(densitas["rendah"], max(sulphates["sedang"], sulphates["tinggi"]), alkohol["tinggi"])

    alpha_buruk = max(r1, r2, r3, r4, r5)
    alpha_standar = max(r6, r7, r8, r9, r10)
    alpha_bagus = max(r11, r12, r13, r14, r15)

    # --- Defuzzifikasi Mamdani (centroid) ---
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

    # --- Defuzzifikasi Sugeno (weighted average, nilai tetap = puncak MF) ---
    tot_Bi_s = alpha_bagus + alpha_buruk + alpha_standar
    sugeno = (
        (alpha_bagus * 8.5) + (alpha_standar * 5.5) + (alpha_buruk * 2.5)
    ) / tot_Bi_s if tot_Bi_s != 0 else 0.0

    return {
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


# =========================================================
# 5. EVALUASI AKURASI PADA DATASET
# =========================================================

@st.cache_data(show_spinner=False)
def evaluasi_dataset(df: pd.DataFrame):
    records = []
    for idx in range(len(df)):
        fa = float(df["fixed acidity"].iloc[idx])
        rs = float(df["residual sugar"].iloc[idx])
        d = float(df["density"].iloc[idx])
        p = float(df["pH"].iloc[idx])
        s = float(df["sulphates"].iloc[idx])
        a = float(df["alcohol"].iloc[idx])
        q = float(df["quality"].iloc[idx])

        hasil = proses_fuzzy(fa, rs, d, p, s, a)

        records.append({
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
        })

    hasil_df = pd.DataFrame(records)
    hasil_df["kategori_asli"] = hasil_df["kualitas_asli"].apply(kategori_quality_asli)

    akurasi_mamdani_kategori = (hasil_df["kategori_mamdani"] == hasil_df["kategori_asli"]).mean() * 100
    akurasi_sugeno_kategori = (hasil_df["kategori_sugeno"] == hasil_df["kategori_asli"]).mean() * 100

    akurasi_mamdani_toleransi = (abs(hasil_df["kualitas_mamdani"] - hasil_df["kualitas_asli"]) <= 1).mean() * 100
    akurasi_sugeno_toleransi = (abs(hasil_df["kualitas_sugeno"] - hasil_df["kualitas_asli"]) <= 1).mean() * 100

    metrik = {
        "akurasi_mamdani_kategori": akurasi_mamdani_kategori,
        "akurasi_sugeno_kategori": akurasi_sugeno_kategori,
        "akurasi_mamdani_toleransi": akurasi_mamdani_toleransi,
        "akurasi_sugeno_toleransi": akurasi_sugeno_toleransi,
    }
    return hasil_df, metrik


# =========================================================
# 6. FUNGSI PLOT
# =========================================================

_WARNA = {
    "rendah": "tab:blue", "sedang": "tab:orange", "tinggi": "tab:red",
    "normal": "tab:green", "asam": "tab:blue", "basa": "tab:red",
    "buruk": "tab:red", "standar": "tab:orange", "bagus": "tab:green",
}


def plot_membership(x_range, curves, judul, xlabel, nilai_input=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for label, y in curves.items():
        ax.plot(x_range, y, label=label.capitalize(), color=_WARNA.get(label), linewidth=2)
    if nilai_input is not None:
        ax.axvline(nilai_input, color="black", linestyle="--", linewidth=1.5,
                   label=f"Input = {nilai_input:.3g}")
    ax.set_title(judul, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("\u03bc(x)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
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


# =========================================================
# 7. STREAMLIT APP
# =========================================================

st.set_page_config(page_title="Fuzzy Wine Quality (Mamdani & Sugeno)", layout="wide")

st.title("\U0001F377 Sistem Fuzzy Mamdani & Sugeno \u2014 Prediksi Kualitas Wine")
st.caption(
    "Gabungan dari notebook `tubes_DKA_ver3.ipynb` (engine fuzzy) dan "
    "`Menghitung_Akurasi.ipynb` (evaluasi akurasi)."
)

tab_prediksi, tab_evaluasi = st.tabs(["\U0001F52E Prediksi Tunggal", "\U0001F4CA Evaluasi Dataset (Akurasi)"])

# ---------------------------------------------------------
# TAB 1 : PREDIKSI TUNGGAL
# ---------------------------------------------------------
with tab_prediksi:
    st.subheader("Masukkan Parameter Wine")
    c1, c2, c3 = st.columns(3)
    with c1:
        fa = st.number_input("Fixed Acidity (0.0 - 20.0)", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
        rs = st.number_input("Residual Sugar (0.0 - 70.0)", min_value=0.0, max_value=70.0, value=6.0, step=0.1)
    with c2:
        d = st.number_input("Density (0.985 - 1.04)", min_value=0.985, max_value=1.04, value=0.995, step=0.001, format="%.4f")
        p = st.number_input("pH (2.5 - 4.2)", min_value=2.5, max_value=4.2, value=3.2, step=0.01)
    with c3:
        s = st.number_input("Sulphates (0.1 - 1.4)", min_value=0.1, max_value=1.4, value=0.5, step=0.01)
        a = st.number_input("Alcohol (%) (7.0 - 16.0)", min_value=7.0, max_value=16.0, value=10.5, step=0.1)

    if st.button("Hitung Kualitas", type="primary"):
        st.session_state["hasil_prediksi"] = proses_fuzzy(fa, rs, d, p, s, a)
        st.session_state["input_prediksi"] = (fa, rs, d, p, s, a)

    if "hasil_prediksi" in st.session_state:
        hasil = st.session_state["hasil_prediksi"]
        fa, rs, d, p, s, a = st.session_state["input_prediksi"]

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
            x_fa = np.linspace(0, 17, 300)
            x_rs = np.linspace(0, 30, 300)
            x_den = np.linspace(0.985, 1.01, 300)
            x_ph = np.linspace(2.5, 4.2, 300)
            x_sul = np.linspace(0.1, 1.4, 300)
            x_al = np.linspace(7, 16, 300)

            g1, g2 = st.columns(2)
            with g1:
                st.pyplot(plot_membership(x_fa, {
                    "rendah": trapez_vec(x_fa, 3, 3, 5, 6),
                    "sedang": trapez_vec(x_fa, 5, 6, 8, 9),
                    "tinggi": trapez_vec(x_fa, 8, 10, 15, 15),
                }, "Fixed Acidity", "Nilai Fixed Acidity", fa))

                st.pyplot(plot_membership(x_den, {
                    "rendah": trapez_vec(x_den, 0.987, 0.987, 0.992, 0.994),
                    "normal": trapez_vec(x_den, 0.992, 0.994, 0.998, 1.0),
                    "tinggi": trapez_vec(x_den, 0.998, 1.0, 1.04, 1.04),
                }, "Density", "Nilai Density", d))

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

# ---------------------------------------------------------
# TAB 2 : EVALUASI DATASET / AKURASI
# ---------------------------------------------------------
with tab_evaluasi:
    st.subheader("Evaluasi Akurasi pada Dataset")
    st.write(
        "Unggah file CSV dengan kolom: `fixed acidity`, `residual sugar`, `density`, "
        "`pH`, `sulphates`, `alcohol`, `quality` (format seperti `winequality-white-indonesia.csv`)."
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

        kolom_perlu = {"fixed acidity", "residual sugar", "density", "pH", "sulphates", "alcohol", "quality"}
        if not kolom_perlu.issubset(set(df.columns)):
            st.error(f"Kolom berikut belum ada di dataset: {kolom_perlu - set(df.columns)}")
        else:
            if st.button("Jalankan Evaluasi", type="primary"):
                with st.spinner("Menghitung fuzzy inference untuk setiap baris..."):
                    hasil_df, metrik = evaluasi_dataset(df)
                st.session_state["hasil_df"] = hasil_df
                st.session_state["metrik"] = metrik

            if "hasil_df" in st.session_state:
                hasil_df = st.session_state["hasil_df"]
                metrik = st.session_state["metrik"]

                st.markdown("### Hasil Akurasi")
                ak1, ak2 = st.columns(2)
                with ak1:
                    st.metric("Akurasi Mamdani (kategori)", f"{metrik['akurasi_mamdani_kategori']:.2f}%")
                    st.metric("Akurasi Mamdani (Kualitas \u00b11)", f"{metrik['akurasi_mamdani_Kualitas']:.2f}%")
                with ak2:
                    st.metric("Akurasi Sugeno (kategori)", f"{metrik['akurasi_sugeno_kategori']:.2f}%")
                    st.metric("Akurasi Sugeno (Kualitas \u00b11)", f"{metrik['akurasi_sugeno_Kualitas']:.2f}%")

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
