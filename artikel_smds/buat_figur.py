"""Membuat gambar dan angka untuk artikel (Python murni: numpy, pandas, matplotlib).
Jalankan dari folder artikel_smds:  python3 buat_figur.py
Menghitung ulang analisis lead-lag (korelasi silang, regresi lead dengan galat Newey-West)
dari ../data dan memeriksanya terhadap hasil R (R/analisis_R.Rmd); gambar pembanding model memakai ../R/hasil_R.json.
Menghasilkan dua set gambar: figs/ (Indonesia) dan figs_en/ (English)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150})
d = pd.read_csv("../data/master_dataset_godzilla_elnino_2000_2025.csv", parse_dates=["date_key"])
d["m"] = d.date_key.dt.month
REG = ["Indo", "Sumatera", "Jawa", "Kalimantan", "Sulawesi", "NusaTenggara", "Maluku", "Papua"]
for r in REG:  # anomali = SDCI dikurangi klimatologi bulanan
    d["A_" + r] = d["SDCI_" + r] - d.groupby("m")["SDCI_" + r].transform("mean")

def nw_ols(y, x, L=12):
    """OLS y~1+x dengan galat Newey-West (kernel Bartlett, lag L, tanpa prewhitening/penyesuaian):
    padanan sandwich::NeweyWest(m, lag=12, prewhite=FALSE)."""
    X = np.c_[np.ones(len(x)), x]; b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    xu = X * u[:, None]; S = xu.T @ xu
    for l in range(1, L + 1):
        G = xu[l:].T @ xu[:-l]; S += (1 - l / (L + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ S @ XtXi
    se = np.sqrt(V[1, 1])
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return b[1], se, r2

cc = {r: [np.corrcoef(d.ONI.values[: len(d) - h], d["A_" + r].values[h:])[0, 1] for h in range(13)] for r in REG}
lead = []
for h in range(10):
    b, se, r2 = nw_ols(d.A_Indo.values[h:], d.ONI.values[: len(d) - h]); lead.append((h, b, se, r2))
assert abs(lead[0][1] - -0.0282) < 5e-4 and abs(lead[0][2] - 0.0122) < 5e-4, "tidak cocok dengan R"  # cek silang R
pd.DataFrame(lead, columns=["h", "beta", "se", "R2"]).round(4).to_csv("tabel_lead.csv", index=False)
pd.DataFrame(cc).round(4).to_csv("tabel_crosscorr.csv", index_label="h")
_R = json.load(open("../R/hasil_R.json"))   # keluaran R/godzila-el-nino.Rmd
_short = {"ARIMA (Statistical Baseline)": "ARIMA(1,1,1)", "Prophet (Meta Time-Series)": "Prophet", "Random Forest (Bagging Baseline)": "Random Forest",
          "LightGBM (Boosting Baseline)": "LightGBM", "XGBoost (Boosting Default)": "XGBoost default", "Klimatologi (rata-rata bulanan)": "Klimatologi"}
_te = {m["Metric"]: m["Test"] for m in _R["metrics"]}
bench = {"XGBoost (tuned)": {"RMSE": _te["RMSE"], "MAE": _te["MAE"], "R2": _te["R2 Score"]}}
bench.update({_short[k]: {"RMSE": v["RMSE"], "MAE": v["MAE"], "R2": v["R2 Score"]} for k, v in _R["bench"].items()})
bench = dict(sorted(bench.items(), key=lambda kv: kv[1]["RMSE"]))

L = {  # label per bahasa
    "id": dict(dir="figs", dec=",", ts_sdci="Anomali SDCI nasional", cc_x="Lead h (bulan)", beta_ci="CI 95% Newey-West", beta_y="Koefisien ONI",
               bench_x=r"$R^2$ uji (2021–2025), lead 3 bulan; sumbu dipotong di $-0{,}5$", clim="Klimatologi", tuned="XGBoost (disetel)",
               leak=["Komponen SDCI\n(NDVI, LST, curah hujan,\nbulan sama)", "ONI + SST\n(bulan sama)", "ENSO lag-3\n+ musiman", "Klimatologi\n(musiman saja)"], leak_y=r"$R^2$ uji", wf=("Latih", "Validasi", "Uji (60 bulan)", "Tahun", "Uji", "Lipatan")),
    "en": dict(dir="figs_en", dec=".", ts_sdci="National SDCI anomaly", cc_x="Lead h (months)", beta_ci="95% Newey-West CI", beta_y="ONI coefficient",
               bench_x=r"Test $R^2$ (2021–2025), 3-month lead; axis clipped at $-0.5$", clim="Climatology", tuned="XGBoost (tuned)",
               leak=["SDCI components\n(NDVI, LST, rainfall,\nsame month)", "ONI + SST\n(same month)", "ENSO lag-3\n+ seasonal", "Climatology\n(seasonal only)"], leak_y=r"Test $R^2$", wf=("Train", "Validation", "Test (60 months)", "Year", "Test", "Fold")),
}
z = lambda v: (v - v.mean()) / v.std()

def make(lang):
    T = L[lang]; os.makedirs(T["dir"], exist_ok=True); sv = lambda fig, n: [fig.savefig(f"{T['dir']}/{n}.{e}") for e in ("pdf", "png")] and plt.close(fig)
    fmt = lambda v, n=3: f"{v:.{n}f}".replace(".", T["dec"]).replace("-", "−")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(d.date_key, z(d.A_Indo), color="#b2182b", lw=1.1, label=T["ts_sdci"]); ax.plot(d.date_key, z(d.ONI), color="#2166ac", lw=1.1, label="ONI")
    ax.axhline(0, color="k", lw=.5); ax.set_ylabel("z-score"); ax.legend(frameon=False, ncol=2, loc="upper left"); fig.tight_layout(); sv(fig, "fig_ts")

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for c, r in zip(plt.cm.tab10(np.arange(8)), REG):
        ax.plot(range(13), cc[r], marker="o", ms=2.5, lw=1.1, color=c, label=r, ls="-" if r != "Indo" else "--")
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel(T["cc_x"]); ax.set_ylabel(r"$\mathrm{corr}(\mathrm{ONI}_t,\,A_{t+h})$")
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="lower right"); fig.tight_layout(); sv(fig, "fig_cc")

    h_, b_, se_ = [np.array(v) for v in zip(*[(a, b, s) for a, b, s, _ in lead])]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.fill_between(h_, b_ - 1.96 * se_, b_ + 1.96 * se_, color="#2166ac", alpha=.2, label=T["beta_ci"])
    ax.plot(h_, b_, color="#2166ac", marker="o", ms=3, label=r"$\hat\beta_h$"); ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel(T["cc_x"]); ax.set_ylabel(T["beta_y"]); ax.legend(frameon=False); fig.tight_layout(); sv(fig, "fig_beta")

    bn = {k: v for k, v in bench.items()}; names = list(bn)[::-1]
    lab = lambda n: T["clim"] if n == "Klimatologi" else (T["tuned"] if n == "XGBoost (tuned)" else n)
    vals = [bn[n]["R2"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.barh([lab(n) for n in names], np.clip(vals, -0.5, None), color=["#b2182b" if n == "Klimatologi" else "#2166ac" if n == "XGBoost (tuned)" else "#999999" for n in names])
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel(T["bench_x"]); ax.set_xlim(-0.56, 0.1)
    for i, v in enumerate(vals):
        t = fmt(v)
        if v >= 0: ax.text(v + 0.008, i, t, va="center", fontsize=7)
        elif v < -0.5: ax.text(-0.49, i, t, va="center", fontsize=7, color="white")
        else: ax.text(v - 0.008, i, t, va="center", ha="right", fontsize=7)
    fig.tight_layout(); sv(fig, "fig_bench")

    fig, ax = plt.subplots(figsize=(6.4, 2.3)); v = [1.0, -0.373, -0.0554, -0.0505]
    ax.bar(T["leak"], v, color=["#b2182b", "#999999", "#2166ac", "#999999"]); ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel(T["leak_y"]); ax.tick_params(axis="x", labelsize=7); ax.set_ylim(-0.6, 1.18)
    for i, x in enumerate(v): ax.text(i, x + (0.04 if x >= 0 else -0.04), fmt(x), ha="center", va="bottom" if x >= 0 else "top", fontsize=7)
    fig.tight_layout(); sv(fig, "fig_leak")

    fig, ax = plt.subplots(figsize=(6.4, 2.6)); t0 = 2001 + 3 / 12  # sampel Apr 2001 - Des 2025; latih 237 bulan; 5 lipatan, validasi 39 bulan
    for i, n in enumerate([42, 81, 120, 159, 198]):
        a, b, c = t0, t0 + n / 12, t0 + (n + 39) / 12; y = 5 - i
        ax.barh(y, b - a, left=a, color="#2166ac", height=.6); ax.barh(y, c - b, left=b, color="#f4a582", height=.6)
        ax.text((a + b) / 2, y, T["wf"][0], ha="center", va="center", color="white", fontsize=7); ax.text((b + c) / 2, y, T["wf"][1], ha="center", va="center", fontsize=7)
    ts = 2021.0; ax.barh(0, 2026 - ts, left=ts, color="#b2182b", height=.6); ax.text((ts + 2026) / 2, 0, T["wf"][2], ha="center", va="center", color="white", fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels([T["wf"][4]] + [f"{T['wf'][5]} {k}" for k in range(5, 0, -1)]); ax.set_xlabel(T["wf"][3]); ax.set_xlim(2001, 2026.2); fig.tight_layout(); sv(fig, "fig_wfv")

for lang in L: make(lang)
print("OK; lead h=0:", [round(x, 4) for x in lead[0]])
