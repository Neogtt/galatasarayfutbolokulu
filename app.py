import io
from datetime import date
import datetime as dt
from typing import Tuple, Optional

import pandas as pd
import streamlit as st


# ==========================
# Yükleme / Kaydetme
# ==========================

@st.cache_data(show_spinner=False)
def load_excel(excel_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Excel içeriğini DataFrame'lere yükler.
    Sütun adları alias ile normalleştirilir; yok sayfa eksikse boş döner.
    """
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    sheets = xls.sheet_names

    def read_sheet(name, cols=None):
        if name in sheets:
            df = xls.parse(name)
        else:
            df = pd.DataFrame(columns=cols or [])
        return df

    # Kullanıcı başlıkları → dahili alanlar
    ALIASES = {
        "ADI SOYADI": "AdSoyad",
        "Adı Soyadı": "AdSoyad",
        "KayıtTarihi": "Baslangic",
        "Uyeliktercihi": "UyelikTercihi",
        "UyelikYenilemeTarihi": "SonOdeme",
        "UyelikgunTercihi": "UyelikGunTercihi",
        "UyelikYenilemetercihi": "UyelikYenilemeTercihi",
    }

    def normalize_students_df(df: pd.DataFrame) -> pd.DataFrame:
        # Alias uygula
        ren = {c: ALIASES.get(str(c), c) for c in df.columns}
        df = df.rename(columns=ren)

        # Beklenen kolonlar
        base_cols = [
            "ID","AdSoyad","DogumTarihi","VeliAdSoyad","Telefon",
            "Grup","Seviye","Koc","Baslangic","UcretAylik","SonOdeme","Aktif",
            "UyelikTercihi","UyelikGunTercihi","UyelikYenilemeTercihi"
        ]
        for c in base_cols:
            if c not in df.columns:
                df[c] = None

        # Tür düzeltmeleri
        df["ID"] = pd.to_numeric(df.get("ID", range(1,len(df)+1)), errors="coerce").fillna(0).astype(int)
        for c in ("DogumTarihi","Baslangic","SonOdeme"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
        df["UcretAylik"] = pd.to_numeric(df.get("UcretAylik", 0), errors="coerce").fillna(0)
        df["Telefon"] = df.get("Telefon","").astype(str)
        df["Aktif"] = df.get("Aktif", True).fillna(True).astype(bool)
        # Üyelik kodlarını (0-4) int olarak tut
        df["UyelikTercihi"] = pd.to_numeric(df.get("UyelikTercihi", 0), errors="coerce").fillna(0).astype(int)

        return df[base_cols]

    # Öğrenciler sayfası: "Ogrenciler" varsa onu, yoksa ilk sayfayı kullan
    if "Ogrenciler" in sheets:
        ogr_raw = xls.parse("Ogrenciler")
    else:
        ogr_raw = xls.parse(sheets[0]) if sheets else pd.DataFrame()

    ogr = normalize_students_df(ogr_raw)

    # Yoklama ve Tahsilat (opsiyonel)
    yok = read_sheet("Yoklama", ["Tarih","Grup","OgrenciID","AdSoyad","Koc","Katildi","Not"])
    tah = read_sheet("Tahsilat", ["Tarih","OgrenciID","AdSoyad","Koc","Tutar","Aciklama"])

    # Tip düzeltmeleri
    for df_, dcol in [(yok,"Tarih"), (tah,"Tarih")]:
        if not df_.empty and dcol in df_.columns:
            df_[dcol] = pd.to_datetime(df_[dcol], errors="coerce").dt.date

    return ogr, yok, tah


def write_excel(ogr: pd.DataFrame, yok: pd.DataFrame, tah: pd.DataFrame) -> bytes:
    """DataFrame'leri tek bir Excel bytes objesine yazar ve döndürür."""
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as w:
        ogr.to_excel(w, index=False, sheet_name="Ogrenciler")
        yok.to_excel(w, index=False, sheet_name="Yoklama")
        tah.to_excel(w, index=False, sheet_name="Tahsilat")
    buff.seek(0)
    return buff.read()


# ==========================
# Uygulama Ayarları
# ==========================

st.set_page_config(page_title="Futbol Okulu", page_icon="⚽", layout="wide")
st.sidebar.title("⚽ Futbol Okulu")
st.sidebar.caption("Excel tabanlı MVP — Üyelik takip + Genel Bakış")

uploaded = st.sidebar.file_uploader("Excel yükle (.xlsx)", type=["xlsx"])
if uploaded:
    ogr, yok, tah = load_excel(uploaded.getvalue())
else:
    # Demo veri
    ogr = pd.DataFrame([
        {"ID":1,"AdSoyad":"Ali Yılmaz","VeliAdSoyad":"Mehmet Yılmaz","Telefon":"05330000000",
         "Grup":"U10","Seviye":"Başlangıç","Koc":"Ahmet","Baslangic":dt.date(2025,9,1),
         "UcretAylik":1500,"SonOdeme":dt.date(2025,10,1),"Aktif":True,"UyelikTercihi":1,
         "UyelikGunTercihi":"1-7","UyelikYenilemeTercihi":"Otomatik"},
        {"ID":2,"AdSoyad":"Efe Demir","VeliAdSoyad":"Aylin Demir","Telefon":"05331111111",
         "Grup":"U10","Seviye":"Orta","Koc":"Ahmet","Baslangic":dt.date(2025,8,15),
         "UcretAylik":1500,"SonOdeme":dt.date(2025,9,1),"Aktif":True,"UyelikTercihi":2,
         "UyelikGunTercihi":"8-15","UyelikYenilemeTercihi":"Manuel"},
        {"ID":3,"AdSoyad":"Berk Kaya","VeliAdSoyad":"Kerem Kaya","Telefon":"05332222222",
         "Grup":"U12","Seviye":"İleri","Koc":"Elif","Baslangic":dt.date(2025,7,1),
         "UcretAylik":1750,"SonOdeme":dt.date(2025,10,1),"Aktif":True,"UyelikTercihi":4,
         "UyelikGunTercihi":"16-31","UyelikYenilemeTercihi":"Manuel"},
    ])
    yok = pd.DataFrame(columns=["Tarih","Grup","OgrenciID","AdSoyad","Koc","Katildi","Not"])
    tah = pd.DataFrame(columns=["Tarih","OgrenciID","AdSoyad","Koc","Tutar","Aciklama"])


# ==========================
# GENEL BAKIŞ PANELİ
# ==========================

st.header("Genel Bakış")

# Üyelik kodu → ay sayısı
UYELIK_AY = {0: 0, 1: 1, 2: 3, 3: 6, 4: 12}

def add_months(d: date, months: int) -> Optional[date]:
    if pd.isna(d) or d is None or months is None:
        return None
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # Ay sonu güvenliği
    mdays = [31, 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28, 31,30,31,30,31,31,30,31,30,31][m-1]
    day = min(d.day, mdays)
    return date(y, m, day)

_today = dt.date.today()

def build_expiry_df(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Baslangic"] = pd.to_datetime(tmp.get("Baslangic"), errors="coerce").dt.date
    tmp["UyelikTercihi"] = pd.to_numeric(tmp.get("UyelikTercihi", 0), errors="coerce").fillna(0).astype(int)

    expires, remain, sure_ay = [], [], []
    for _, r in tmp.iterrows():
        code = int(r.get("UyelikTercihi", 0))
        months = UYELIK_AY.get(code, 0)
        sure_ay.append(months)
        start = r.get("Baslangic")
        e = add_months(start, months) if months > 0 else None
        expires.append(e)
        remain.append(None if e is None else (e - _today).days)

    tmp["UyelikSuresiAy"] = sure_ay
    tmp["UyelikBitisTarihi"] = expires
    tmp["KalanGun"] = remain

    cols = ["ID","AdSoyad","Koc","Grup","Baslangic","UyelikTercihi","UyelikSuresiAy","UyelikBitisTarihi","KalanGun","Telefon"]
    return tmp[cols].sort_values(["KalanGun"], ascending=True, na_position="last")

exp_df = build_expiry_df(ogr)

# KPI'lar
aktif_say = int(ogr.get("Aktif", pd.Series([True]*len(ogr))).sum()) if not ogr.empty else 0
uyelikli = int((exp_df["UyelikSuresiAy"] > 0).sum()) if not exp_df.empty else 0
bugun_icinde = int((exp_df["KalanGun"] == 0).sum()) if not exp_df.empty else 0
hafta_icinde = int(((exp_df["KalanGun"] >= 0) & (exp_df["KalanGun"] <= 5)).sum()) if not exp_df.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aktif Öğrenci", aktif_say)
c2.metric("Üyelikli Öğrenci", uyelikli)
c3.metric("Bugün Biten", bugun_icinde)
c4.metric("≤5 Gün Kalan", hafta_icinde)

st.markdown("**≤ 5 gün kalan üyelikler**")
if not exp_df.empty and ((exp_df["KalanGun"] >= 0) & (exp_df["KalanGun"] <= 5)).any():
    st.dataframe(exp_df.loc[(exp_df["KalanGun"] >= 0) & (exp_df["KalanGun"] <= 5)], use_container_width=True, hide_index=True)
else:
    st.info("Önümüzdeki 5 gün içinde biten üyelik bulunmuyor.")

st.markdown("**Tüm öğrenciler — Üyelik bitiş ve kalan gün**")
st.dataframe(exp_df, use_container_width=True, hide_index=True)

st.divider()


# ==========================
# ÖĞRENCİ EKLE
# ==========================

st.header("Öğrenci Ekle")

uyelik_map = {
    "Kontenjan (0)": 0,
    "1 Aylık (1)": 1,
    "3 Aylık (2)": 2,
    "6 Aylık (3)": 3,
    "12 Aylık (4)": 4,
}

col1, col2, col3 = st.columns(3)
ad = col1.text_input("Ad Soyad")
veli = col2.text_input("Veli Adı")
tel = col3.text_input("Telefon")

col4, col5, col6 = st.columns(3)
gr = col4.text_input("Grup", "U10")
sev = col5.selectbox("Seviye", ["Başlangıç","Orta","İleri"])
koc = col6.text_input("Koç")

col7, col8, col9 = st.columns(3)
kayit = col7.date_input("Kayıt Tarihi", dt.date.today())
ucret = col8.number_input("Aylık Ücret (TL)", 0, 100000, 1500, 50)
aktif = col9.toggle("Aktif", True)

col10, col11, col12 = st.columns(3)
uyelik = col10.selectbox("Üyelik Tercihi", list(uyelik_map.keys()))
uyelik_gun = col11.text_input("Üyelik Gün Tercihi", "1-7")
uyelik_yen = col12.text_input("Üyelik Yenileme Tercihi", "Otomatik")

if st.button("➕ Ekle") and ad:
    new_id = (ogr["ID"].max() + 1) if not ogr.empty else 1
    new_row = {
        "ID": new_id, "AdSoyad": ad, "VeliAdSoyad": veli, "Telefon": tel,
        "Grup": gr, "Seviye": sev, "Koc": koc,
        "Baslangic": kayit, "UcretAylik": ucret, "SonOdeme": dt.date.today(), "Aktif": aktif,
        "UyelikTercihi": uyelik_map[uyelik], "UyelikGunTercihi": uyelik_gun, "UyelikYenilemeTercihi": uyelik_yen,
    }
    ogr = pd.concat([ogr, pd.DataFrame([new_row])], ignore_index=True)
    st.success(f"{ad} eklendi — Üyelik kodu: {uyelik_map[uyelik]}")

st.subheader("Öğrenci Listesi")
st.dataframe(ogr, use_container_width=True, hide_index=True)

# ==========================
# Dışa Aktar
# ==========================

def write_requirements() -> str:
    return "streamlit>=1.34\npandas>=2.0\nopenpyxl>=3.1\n"

excel_bytes = write_excel(ogr, yok, tah)
st.download_button("Excel'i indir", data=excel_bytes,
                   file_name=f"futbol_okulu_{dt.date.today().isoformat()}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Py3.8+ uyumlu sürüm — Genel Bakış + Üyelik kodları (0-4).")

