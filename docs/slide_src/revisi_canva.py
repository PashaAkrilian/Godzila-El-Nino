"""Merevisi Slide_Presentasi (ekspor PDF dari Canva) TANPA menyentuh Canva.
Judul, hiasan, dan latar asli dipertahankan; hanya tabel/grafik/angka yang tidak sesuai hasil terkoreksi yang diganti.
Input : ../arsip/Slide_Presentasi_asli_canva.pdf, ../../R/hasil_R.json (keluaran R/godzila-el-nino.Rmd, seed 42)
Output: ../Kelompok 1_Presentasi.pdf
Jalankan dari folder ini:  python revisi_canva.py   (butuh pymupdf, matplotlib, pandas, numpy)"""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
warnings.filterwarnings("ignore")
import pymupdf as fitz

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, OUT = os.path.join(HERE, "../arsip/Slide_Presentasi_asli_canva.pdf"), os.path.join(HERE, "../Kelompok 1_Presentasi.pdf")
FONT = {k: os.path.join(HERE, f"fonts/Poppins-{k}.ttf") for k in ("Regular", "Bold", "SemiBold")}
for f in FONT.values(): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Poppins"
R = json.load(open(os.path.join(HERE, "../../R/hasil_R.json")))
TMP = os.path.join(HERE, "_tmp"); os.makedirs(TMP, exist_ok=True)
BLUE, YEL, GREYB, RED = "#0B5FFF", "#FFE033", "#4a4a4a", "#B2182B"
fnum = lambda v, n=4: f"{v:.{n}f}".replace("-", "−")   # titik desimal seperti slide asli, minus tipografis

def canvas(w_pt, h_pt):
    fig = plt.figure(figsize=(w_pt / 72, h_pt / 72), dpi=216); fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w_pt); ax.set_ylim(h_pt, 0); ax.axis("off"); ax.patch.set_alpha(0)
    return fig, ax

def box(ax, x, y, w, h, fill, edge=None, r=6, lw=1.4):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}", fc=fill, ec=edge or fill, lw=lw))

def table_img(path, w, h, headers, rows, colw, hl_rows=(), fs=19, hfs=17, gap=6, footer=None, first_bold=False, pad_top=0):
    fig, ax = canvas(w, h)
    n = len(rows) + 1 + (1 if footer else 0); rh = (h - pad_top - gap * (n - 1)) / n
    xs = np.r_[0, np.cumsum(colw)]; scale = (w - gap * (len(colw) - 1)) / sum(colw)
    cw = [c * scale for c in colw]; xs = [sum(cw[:i]) + gap * i for i in range(len(cw))]
    for j, t in enumerate(headers):
        box(ax, xs[j], pad_top, cw[j], rh, BLUE); ax.text(xs[j] + cw[j] / 2, pad_top + rh / 2, t, color="white", ha="center", va="center", fontsize=hfs, fontweight="bold")
    for i, row in enumerate(rows):
        y = pad_top + (i + 1) * (rh + gap)
        for j, t in enumerate(row):
            box(ax, xs[j], y, cw[j], rh, "#F2F2F2", GREYB)
            if i in hl_rows: box(ax, xs[j] + 10, y + 9, cw[j] - 20, rh - 18, YEL, YEL, r=5)
            ax.text(xs[j] + cw[j] / 2, y + rh / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if (i in hl_rows or (first_bold and j == 0)) else "regular", color="black")
    if footer: ax.text(w / 2, h - rh / 2, footer, ha="center", va="center", fontsize=hfs - 1, color=GREYB, style="italic")
    fig.savefig(path, transparent=True); plt.close(fig)

# ------------------------------------------------------------------ gambar baru
bench = R["bench"]; m = {r["Metric"]: r for r in R["metrics"]}
order = [("ARIMA", "ARIMA (Statistical Baseline)"), ("Prophet", "Prophet (Meta Time-Series)"), ("Random Forest", "Random Forest (Bagging Baseline)"),
         ("LightGBM", "LightGBM (Boosting Baseline)"), ("Klimatologi", "Klimatologi (rata-rata bulanan)")]
rows6 = [[n, fnum(bench[k]["RMSE"]), fnum(bench[k]["MAE"]), fnum(bench[k]["R2 Score"])] for n, k in order]
rows6.append(["XGBoost", fnum(m["RMSE"]["Test"]), fnum(m["MAE"]["Test"]), fnum(m["R2 Score"]["Test"])])
table_img(f"{TMP}/t6.png", 728, 471, ["MODEL", "RMSE", "MAE", "R-SQUARED (R²)"], rows6, [2.3, 1.2, 1.2, 1.7], hl_rows=(5,), fs=17, hfs=15, gap=5)

bp = R["best_params"]
rows9 = [[k, str(bp[k]) if isinstance(bp[k], int) else f"{bp[k]:.5f}"] for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "lambda", "alpha", "gamma"]]
table_img(f"{TMP}/t9.png", 686, 548, ["HYPERPARAMETER", "NILAI TERBAIK"], rows9, [1.0, 1.0], fs=17, hfs=15, gap=5, first_bold=True, pad_top=48,
          footer=f"RMSE validasi walk-forward terbaik {R['best_value']:.4f}  ·  percobaan #{R['best_trial']} dari 50 (random search)")

F = R["fold"]
_d = pd.read_csv(os.path.join(HERE, "../../data/master_dataset_godzilla_elnino_2000_2025.csv"), parse_dates=["date_key"])
_tr = _d[_d.date_key >= "2001-04-01"].reset_index(drop=True); _tr = _tr.iloc[: int(len(_tr) * .8)]
_y, _mo = _tr.SDCI_Indo.values, _tr.date_key.dt.month.values
def _klim_r2(n, v=39):  # R² klimatologi bulanan pada fold walk-forward yang sama (latih n bulan, validasi 39 bulan)
    c = {mm: _y[:n][_mo[:n] == mm].mean() for mm in range(1, 13)}; p = np.array([c[x] for x in _mo[n:n + v]]); a = _y[n:n + v]
    return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()
klim = {i + 1: _klim_r2(n) for i, n in enumerate([42, 81, 120, 159, 198])}
rows11a = [[f["Fold"], fnum(f["RMSE"]), fnum(f["MAE"]), fnum(f["R2 Score"]), fnum(klim[i + 1], 3)] for i, f in enumerate(F)]
table_img(f"{TMP}/t11a.png", 712, 230, ["Iterasi", "RMSE", "MAE", "R² XGBoost", "R² klimatologi"], rows11a, [1.1, 1.0, 1.0, 1.4, 1.7], fs=15, hfs=13, gap=3, pad_top=40)
rows11b = [["RMSE", fnum(m["RMSE"]["Train"]), fnum(m["RMSE"]["Test"])], ["MAE", fnum(m["MAE"]["Train"]), fnum(m["MAE"]["Test"])], ["R-Squared (R²)", fnum(m["R2 Score"]["Train"]), fnum(m["R2 Score"]["Test"])]]
table_img(f"{TMP}/t11b.png", 601, 316, ["Metrik Evaluasi", "Data Pelatihan (Train)", "Data Uji (Test)"], rows11b, [1.2, 1.4, 1.2], fs=18, hfs=14, gap=5, first_bold=True, pad_top=40)

# Forecasting 2026
fig, ax = canvas(1020, 561); ax.axis("on"); fig.subplots_adjust(0, 0, 1, 1); ax.set_position([0.075, 0.13, 0.9, 0.74]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.cla(); ax.patch.set_alpha(0)
fd = pd.to_datetime(R["fut_dates"]); x = np.arange(12)
clim_m = pd.read_csv(os.path.join(HERE, "../../data/master_dataset_godzilla_elnino_2000_2025.csv"), parse_dates=["date_key"])
clim_m = clim_m.groupby(clim_m.date_key.dt.month).SDCI_Indo.mean().values
ax.fill_between(x, R["fut_low"], R["fut_high"], color="#9EC8EE", alpha=.55, label="Interval 10%–90%")
ax.plot(x, R["fut_med"], color="#0B5FA8", lw=3, marker="o", ms=5, label="Median prediksi XGBoost")
ax.plot(x, clim_m, color="#555555", lw=2, ls="--", label="Klimatologi 2000–2025")
ax.axhline(0.3, color=RED, ls=":", lw=2, label="Ambang kritis 0,3")
k = int(np.argmin(R["fut_med"])); ax.annotate(f"Sep 2026: {R['fut_med'][k]:.3f}".replace(".", ",") + f"\n({R['fut_low'][k]:.2f}–{R['fut_high'][k]:.2f})".replace(".", ","), (k, R["fut_med"][k]), (k - 3.3, 0.33), fontsize=12, arrowprops=dict(arrowstyle="->"), fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([d.strftime("%b") for d in fd], fontsize=12); ax.set_ylim(0.25, 0.85); ax.tick_params(labelsize=12)
ax.set_ylabel("Skor SDCI nasional", fontsize=13, fontweight="bold"); ax.grid(alpha=.25)
ax.set_title("Skenario 2026: ONI naik 0,15 per bulan (asumsi, bukan prakiraan ONI)", fontsize=14, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=11, frameon=True, ncol=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(f"{TMP}/t12.png", transparent=True); plt.close(fig)

def hbar(path, w, h, data, color, xlabel, title):
    fig, ax = canvas(w, h); ax.axis("on"); ax.set_position([0.40, 0.11, 0.55, 0.80]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.cla(); ax.patch.set_alpha(0)
    names, vals = list(data)[::-1], list(data.values())[::-1]
    ax.barh(range(len(vals)), vals, color=color, edgecolor="#333", lw=.6); ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12); ax.tick_params(axis="x", labelsize=11); ax.grid(axis="x", alpha=.3); ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    for i, v in enumerate(vals): ax.text(v, i, f" {v:.3f}", va="center", fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.18); fig.savefig(path, transparent=True); plt.close(fig)
hbar(f"{TMP}/t13.png", 582, 693, R["shap"], "#1f77d4", "mean(|SHAP|) pada data uji", "Kontribusi fitur global (SHAP)")
hbar(f"{TMP}/t14.png", 746, 593, R["importance"], "#168a8a", "Skor kepentingan relatif (XGBoost)", "15 fitur teratas")

# Walk-forward: sampel 2001-04 .. 2025-12 (297 bulan), latih 237 bulan, 5 lipatan (validasi 39 bulan), uji 60 bulan
fig, ax = canvas(1182, 580); ax.axis("on"); ax.set_position([0.13, 0.17, 0.84, 0.68]); ax.cla(); ax.patch.set_alpha(0)
t0 = pd.Period("2001-04", "M"); yr = lambda p: (p.year + (p.month - 1) / 12)
for i, n in enumerate([42, 81, 120, 159, 198]):
    ys = 5 - i; a = yr(t0); b = yr(t0 + n); c = yr(t0 + n + 39)
    ax.barh(ys, b - a, left=a, color="#1f77d4", height=.6); ax.barh(ys, c - b, left=b, color="#F5A623", height=.6)
    ax.text((a + b) / 2, ys, "Train", ha="center", va="center", color="white", fontsize=12, fontweight="bold"); ax.text((b + c) / 2, ys, "Val", ha="center", va="center", fontsize=12, fontweight="bold")
ts, te = yr(pd.Period("2021-01", "M")), yr(pd.Period("2026-01", "M"))
ax.barh(0, te - ts, left=ts, color="#D62728", height=.6); ax.text((ts + te) / 2, 0, "Uji (60 bulan)", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
ax.axvline(ts, color="#D62728", ls="--", lw=1.5); ax.set_yticks(range(6)); ax.set_yticklabels(["Hidden Test\n(2021–2025)", "Fold-5", "Fold-4", "Fold-3", "Fold-2", "Fold-1"], fontsize=13)
ax.set_xlim(2001, 2026.2); ax.set_xticks(range(2002, 2027, 2)); ax.tick_params(axis="x", labelsize=12); ax.set_xlabel("Tahun Pengamatan (T)", fontsize=13); ax.grid(axis="x", alpha=.25)
ax.set_title("TimeSeriesSplit(5) di dalam data latih (Apr 2001–Des 2020); data uji tidak disentuh", fontsize=13, pad=8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(f"{TMP}/t10.png", transparent=True); plt.close(fig)

# ------------------------------------------------------------------ edit PDF
doc = fitz.open(SRC); NONE, REMOVE = fitz.PDF_REDACT_IMAGE_NONE, fitz.PDF_REDACT_IMAGE_REMOVE
def swap_images(i, mapping):
    """mapping: {rect_lama: png}. Hapus gambar lama lalu pasang gambar baru (transparan) pada kotak yang sama."""
    p = doc[i]
    for r, _ in mapping: p.add_redact_annot(fitz.Rect(*r))
    p.apply_redactions(images=REMOVE, graphics=0, text=fitz.PDF_REDACT_TEXT_NONE)  # hanya gambar; judul tetap
    for r, png in mapping: p.insert_image(fitz.Rect(*r), filename=png)
def swap_text(i, rect, text, font, size, color=(0, 0, 0), baseline_k=1.05):
    p = doc[i]; r = fitz.Rect(*rect); p.add_redact_annot(r); p.apply_redactions(images=NONE, graphics=0)
    p.insert_text((r.x0, r.y0 + size * baseline_k), text, fontsize=size, fontname="pp", fontfile=FONT[font], color=color)

swap_text(4, (907, 490, 970, 560), "40", "Bold", 50)                                   # slide 5: 41 -> 40 variabel numerik
swap_text(6, (1249, 594, 1321, 670), "44", "Bold", 54.4)                                # slide 7: 72 -> 44 fitur
swap_text(6, (967, 254, 1416, 303), "", "Bold", 35)                                                      # slide 7: label "Dikeluarkan (kebocoran)" dihapus
swap_text(6, (1031, 303, 1360, 352), "NDVI, LST, Precip", "Bold", 35, (0.70, 0.09, 0.19))
swap_text(8, (100, 160, 1045, 245), "Pencarian Hiperparameter", "Bold", 56, (0.043, 0.039, 0.039))                 # slide 9: judul tanpa "Optuna" (R tidak punya Optuna)
swap_images(5, [((356, 247, 1084, 718), f"{TMP}/t6.png")])
swap_images(8, [((377, 215, 1063, 763), f"{TMP}/t9.png")])
swap_images(9, [((131, 207, 1313, 787), f"{TMP}/t10.png")])
swap_images(10, [((43, 376, 755, 606), f"{TMP}/t11a.png"), ((780, 364, 1381, 680), f"{TMP}/t11b.png")])
swap_images(11, [((210, 204, 1230, 765), f"{TMP}/t12.png")])
swap_images(12, [((187, 108, 769, 801), f"{TMP}/t13.png")])
swap_images(13, [((138, 136, 884, 729), f"{TMP}/t14.png")])

# slide 15: kesimpulan (teks putih dengan bayangan hitam seperti aslinya)
p = doc[14]; p.add_redact_annot(fitz.Rect(285, 212, 1160, 785)); p.apply_redactions(images=NONE, graphics=0)
QUOTE1 = "“Prediction is very difficult, especially if it's about the future.” — Niels Bohr."
BODY = ("Hasil kami menunjukkan hal serupa. Tanpa komponen penyusun SDCI, XGBoost hanya sedikit lebih baik daripada klimatologi (R² uji 0,070 vs −0,051), "
        "sehingga ENSO belum terbukti mampu memprediksi kekeringan nasional tiga bulan ke depan. Pengaruh ONI justru terasa pada bulan yang sama, "
        "terutama di Jawa dan Nusa Tenggara. Pada skenario ONI terus naik, September 2026 cenderung lebih kering dari biasanya (SDCI 0,399; rentang 0,36–0,54; "
        "rata-rata historis 0,49), tetapi masih di atas ambang kritis 0,3. Menyadari batas ini adalah langkah awal untuk bersiap, karena")
QUOTE2 = "“The first step toward change is awareness.” — Nathaniel Branden."
def put(rect, text, font, size, color):
    return p.insert_textbox(fitz.Rect(*rect), text, fontsize=size, fontname="pp" + font[:1], fontfile=FONT[font], color=color, align=0)
# paragraf rata kanan-kiri; kutipan penutup menyambung langsung setelah "karena"
CSS = ("@font-face {font-family: pp; src: url(Poppins-Regular.ttf);} @font-face {font-family: pp; src: url(Poppins-Bold.ttf); font-weight: bold;}"
       "p {font-family: pp; font-size: 22px; line-height: 1.3; text-align: justify; margin: 0;}")
ARCH = fitz.Archive(os.path.join(HERE, "fonts"))
for dx, dy, col in [(3, 3, (0, 0, 0)), (2, 2, (0, 0, 0)), (1, 1, (0, 0, 0)), (0, 0, (1, 1, 1))]:
    put((296 + dx, 218 + dy, 1150 + dx, 300 + dy), QUOTE1, "Bold", 21, col)
    css = CSS + "p {color: #%02x%02x%02x;}" % tuple(int(255 * c) for c in col)
    spare, _ = p.insert_htmlbox(fitz.Rect(296 + dx, 292 + dy, 1150 + dx, 700 + dy), f"<p>{BODY} <b>{QUOTE2}</b></p>", css=css, archive=ARCH, scale_low=1)
    assert spare >= 0, "teks kesimpulan tidak muat"

doc.save(OUT, garbage=3, deflate=True); print("saved", OUT, len(doc), "halaman")
import subprocess, sys; subprocess.run([sys.executable, os.path.join(HERE, "hapus_logo_unity.py")], check=True)  # logo UNITY dihapus dari slide 1 dan 16
