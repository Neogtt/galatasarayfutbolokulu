"""
FUTBOL OKULU — Streamlit (Py3.8+) — OTOMATIK BAŞLIK TESPİTİ
-----------------------------------------------------------
• Sayfanın adı ne olursa olsun çalışır (örn. 'students').
• Başlık satırı ilk satır değilse otomatik tespit eder (ilk 30 satırı tarar).
• Türkçe karakter/boşluk normalize + geniş alias + header tespiti → Öğrenci Listesi dolar.
• Üyelik kodları: 0=Kontenjan, 1=1 Aylık, 2=3 Aylık, 3=6 Aylık, 4=12 Aylık.
"""

import io
from datetime import date
import datetime as dt
from typing import Tuple, Optional, List

import pandas as pd
import streamlit as st


# ==========================
# Normalizasyon
# ==========================

TR_MAP = str.maketrans({
    "Ç":"c","ç":"c","Ğ":"g","ğ":"g","İ":"i","I":"i","ı":"i","Ö":"o","ö":"o","Ş":"s","ş":"s","Ü":"u","ü":"u"
})

def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().translate(TR_MAP).lower()
    for ch in ["-", ".", "/", "\\", "|"]:
        s = s.replace(ch, " ")
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s

BASE_COLS = [
    "ID","AdSoyad","DogumTarihi","VeliAdSoyad","Telefon",
    "Grup","Seviye","Koc","Baslangic","UcretAylik","SonOdeme","Aktif",
    "UyelikTercihi","UyelikGunTercihi","UyelikYenilemeTercihi"
]

ALIAS_MAP = {
    "id": "ID",
    "adi soyadi": "AdSoyad", "ad soyad": "AdSoyad", "adi": "AdSoyad",
    "dogumt": "DogumTarihi", "dogum t": "DogumTarihi", "dogum tarihi": "DogumTarihi", "dogumtarihi": "DogumTarihi",
    "veliadsoyad": "VeliAdSoyad", "veli ad soyad": "VeliAdSoyad", "veliad": "VeliAdSoyad", "veliadsoy": "VeliAdSoyad",
    "telefon": "Telefon", "tel": "Telefon",
    "grup": "Grup",
    "seviye": "Seviye",
    "koc": "Koc", "koç": "Koc",
    "kayit tarihi": "Baslangic", "kayittarihi": "Baslangic", "kayıt tarihi": "Baslangic", "kayıttarihi": "Baslangic",
    "ucret aylik": "UcretAylik", "ucretaylik": "UcretAylik",
    "son odeme": "SonOdeme", "sonodeme": "SonOdeme",
    "aktif": "Aktif",
    "uyeliktercihi": "UyelikTercihi", "uyelik tercihi": "UyelikTercihi",
    "uyelikyenilemetarihi": "SonOdeme", "uyelik yenileme tarihi": "SonOdeme",
    "uyelikguntercihi": "UyelikGunTercihi", "uyelik gun tercihi": "UyelikGunTercihi",
    "uyelikyenilemetercihi": "UyelikYenilemeTercihi", "uyelik yenileme tercihi": "UyelikYenilemeTercihi",
}

HEADER_HINTS = {"adi","soyadi","telefon","grup","koc","kayit","uyelik","seviye"}


def detect_header_row(df: pd.DataFrame, scan_rows: int = 30) -> int:
    """İlk scan_rows satırı tarar; en çok header ipucu içeren satırı başlık kabul eder.
    Dönen değer: header satır index'i. Bulamazsa 0 döner.
    """
    best_row, best_score = 0, -1
    max_r = min(scan_rows, len(df))
    for r in range(max_r):
        row_vals = [norm_key(v) for v in df.iloc[r].tolist()]
        score = 0
        for v in row_vals:
            for hint in HEADER_HINTS:
                if hint in v:
                    score += 1
        if score > best_score:
            best_score, best_row = score, r
    return best_row


def apply_aliases(cols: List[str]) -> List[str]:
    renamed = []
    for c in cols:
        nk = norm_key(c)
        renamed.append(ALIAS_MAP.get(nk, c))
    return renamed


def normalize_students_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Başlığı tespit et
    hdr = detect_header_row(df)
    header_vals = df.iloc[hdr].tolist()
    # 2) Yeni başlıkları uygula ve üst satırları at
    df2 = df.iloc[hdr+1:].reset_index(drop=True).copy()
    df2.columns = header_vals
    # 3) Alias uygula
    df2 = df2.rename(columns={c: ALIAS_MAP.get(norm_key(c), c) for c in df2.columns})
    # 4) Eksik kolonları tamamla
    for c in BASE_COLS:
        if c not in df2.columns:
            df2[c] = None
    # Türler
    if "ID" in df2.columns:
        df2["ID"] = pd.to_numeric(df2["ID"], errors="coerce")
        if df2["ID"].isna().all():
            df2["ID"] = range(1, len(df2) + 1)
    else:
        df2["ID"] = range(1, len(df2) + 1)
    df2["ID"] = df2["ID"].fillna(0).astype(int)

    for c in ("DogumTarihi","Baslangic","SonOdeme"):
        df2[c] = pd.to_datetime(df2.get(c), errors="coerce").dt.date

    df2["UcretAylik"] = pd.to_numeric(df2.get("UcretAylik", 0), errors="coerce").fillna(0)
    df2["Telefon"] = df2.get("Telefon","").astype(str)

    df2["Aktif"] = df2.get("Aktif", True)
    if not pd.api.types.is_bool_dtype(df2["Aktif"]):
        df2["Aktif"] = df2["Aktif"].fillna(True).astype(bool)

    df2["UyelikTercihi"] = pd.to_numeric(df2.get("UyelikTercihi", 0), errors="coerce").fillna(0).astype(int)
    df2["UyelikTercihi"] = df2["UyelikTercihi"].clip(lower=0, upper=12)  # esneklik, sonra 0-4'e kırparız

    return df2[BASE_COLS]


# ==========================
# Excel Yükleme (header tespitli)
# ==========================

@st.cache_data(show_spinner=False)
def load_excel(excel_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    sheets = xls.sheet_names
    # 'Ogrenciler' tercih et, yoksa ilk sayfa
    sheet = "Ogrenciler" if "Ogrenciler" in sheets else (sheets[0] if sheets else None)
    if sheet is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""
    # Başlıksız (header=None) okuyup tespit yapacağız
    raw = xls.parse(sheet, header=None)
    ogr = normalize_students_df(raw)
    # Opsiyonel sayfalar
    yok = xls.parse("Yoklama") if "Yoklama" in sheets else pd.DataFrame(columns=["Tarih","Grup","OgrenciID","AdSoyad","Koc","Katildi","Not"])
    tah = xls.parse("Tahsilat") if "Tahsilat" in sheets else pd.DataFrame(columns=["Tarih","OgrenciID","AdSoyad","Koc","Tutar","Aciklama"])
    for df_, dcol in [(yok,"Tarih"), (tah,"Tarih")]:
        if not df_.empty and dcol in df_.columns:
            df_[dcol] = pd.to_datetime(df_[dcol], errors="coerce").dt.date
    return ogr, yok, tah, sheet


def write_excel(ogr: pd.DataFrame, yok: pd.DataFrame, tah: pd.DataFrame) -> bytes:
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as w:
        ogr.to_excel(w, index=False, sheet_name="Ogrenciler")
        yok.to_excel(w, index=False, sheet_name="Yoklama")
        tah.to_excel(w, index=False, sheet_name="Tahsilat")
    buff.seek(0)
    return buff.read()


# ==========================
# Uygulama
# ==========================

st.set_page_config(page_title="Futbol Okulu", page_icon="⚽", layout="wide")
st.sidebar.title("⚽ Futbol Okulu — Otomatik Başlık")

uploaded = st.sidebar.file_uploader("Excel yükle (.xlsx)", type=["xlsx"])
if uploaded:
    ogr, yok, tah, used_sheet = load_excel(uploaded.getvalue())
    st.sidebar.success(f"Yüklenen sayfa: {used_sheet}")
else:
    ogr = pd.DataFrame([
        {"ID":1,"AdSoyad":"Demo Öğrenci","Telefon":"0533","Grup":"U10","Seviye":"Başlangıç","Koc":"Ahmet",
         "Baslangic":dt.date(2025,9,1),"UcretAylik":1500,"SonOdeme":dt.date(2025,10,1),"Aktif":True,"UyelikTercihi":1}
    ])
    yok = pd.DataFrame(columns=["Tarih","Grup","OgrenciID","AdSoyad","Koc","Katildi","Not"])
    tah = pd.DataFrame(columns=["Tarih","OgrenciID","AdSoyad","Koc","Tutar","Aciklama"])

# ==========================
# Genel Bakış
# ==========================

st.header("Genel Bakış")

UYELIK_AY = {0:0, 1:1, 2:3, 3:6, 4:12}

def add_months(d: date, months: int) -> Optional[date]:
    if pd.isna(d) or d is None or months is None:
        return None
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    mdays = [31, 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28, 31,30,31,30,31,31,30,31,30,31][m-1]
    day = min(d.day, mdays)
    return date(y, m, day)

_today = dt.date.today()

def build_expiry_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["Baslangic"] = pd.to_datetime(tmp.get("Baslangic"), errors="coerce").dt.date
    tmp["UyelikTercihi"] = pd.to_numeric(tmp.get("UyelikTercihi", 0), errors="coerce").fillna(0).astype(int)
    tmp["UyelikTercihi"] = tmp["UyelikTercihi"].clip(lower=0, upper=4)

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
    for c in cols:
        if c not in tmp.columns:
            tmp[c] = None
    return tmp[cols].sort_values(["KalanGun"], ascending=True, na_position="last")

exp_df = build_expiry_df(ogr)

c1, c2, c3, c4 = st.columns(4)
aktif_say = int(ogr.get("Aktif", pd.Series([True]*len(ogr))).sum()) if not ogr.empty else 0
uyelikli = int((exp_df.get("UyelikSuresiAy", pd.Series([])) > 0).sum()) if not exp_df.empty else 0
bugun_icinde = int((exp_df.get("KalanGun", pd.Series([])) == 0).sum()) if not exp_df.empty else 0
hafta_icinde = int(((exp_df.get("KalanGun", pd.Series([])) >= 0) & (exp_df.get("KalanGun", pd.Series([])) <= 5)).sum()) if not exp_df.empty else 0
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

st.subheader("Öğrenci Listesi")
st.dataframe(ogr, use_container_width=True, hide_index=True)

# ==========================
# Dışa Aktar
# ==========================

def write_excel(ogr, yok, tah):
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as w:
        ogr.to_excel(w, index=False, sheet_name="Ogrenciler")
        yok.to_excel(w, index=False, sheet_name="Yoklama")
        tah.to_excel(w, index=False, sheet_name="Tahsilat")
    buff.seek(0)
    return buff.read()

excel_bytes = write_excel(ogr, yok, tah)
st.download_button("Excel'i indir", data=excel_bytes,
                   file_name=f"futbol_okulu_{dt.date.today().isoformat()}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Otomatik başlık tespiti aktif. Sheet adı: fark etmez (örn. 'students').")
