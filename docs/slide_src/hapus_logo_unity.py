"""Hapus logo "UNITY #14" dari slide 1 dan 16, pertahankan lencana UNNES.
Kotak putih berujung bulat yang tadinya memuat dua logo dipersempit sebatas lencana UNNES.
Pakai:  python hapus_logo_unity.py [masuk.pdf] [keluar.pdf]   (butuh pymupdf)
Default: mengubah ../Slide_Presentasi.pdf menjadi ../Slide_Presentasi.pdf (simpan cadangan dulu).
Dibuat untuk ekspor Canva dengan tata letak ini: gambar logo 400x157 px, lencana 180x240 px."""
import os, sys
import pymupdf as fitz

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../Slide_Presentasi.pdf")
OUT = sys.argv[2] if len(sys.argv) > 2 else SRC
PAGES = (0, 15)  # slide 1 dan slide 16
UNITY_PX, CREST_PX = (400, 157), (180, 240)
PAD_RIGHT = 38   # jarak kotak putih di kanan lencana, pt


def pixmap_with_alpha(doc, xref):
    pix = fitz.Pixmap(doc, xref)
    smask = doc.xref_get_key(xref, "SMask")
    if smask[0] == "xref":
        pix = fitz.Pixmap(pix, fitz.Pixmap(doc, int(smask[1].split()[0])))
    return pix


doc = fitz.open(SRC)
for pno in PAGES:
    p = doc[pno]
    imgs = {(i["width"], i["height"]): i for i in p.get_image_info(xrefs=True) if fitz.Rect(i["bbox"]).y0 < 300 and fitz.Rect(i["bbox"]).x0 < 450}
    if UNITY_PX not in imgs or CREST_PX not in imgs:
        print(f"halaman {pno + 1}: logo tidak ditemukan, dilewati"); continue
    unity, crest = imgs[UNITY_PX], imgs[CREST_PX]
    crest_rect, unity_rect = fitz.Rect(crest["bbox"]), fitz.Rect(unity["bbox"])
    crest_pix = pixmap_with_alpha(doc, crest["xref"])

    # kotak putih lama: persegi panjang putih yang melingkupi kedua logo
    cands = [d["rect"] for d in p.get_drawings() if d.get("fill") == (1.0, 1.0, 1.0) and d["rect"].contains(unity_rect) and d["rect"].contains(crest_rect)]
    pill = min(cands, key=lambda r: r.width * r.height, default=None)  # yang terkecil, bukan latar putih satu halaman
    if pill is None or pill.width > 700 or pill.height > 250:
        print(f"halaman {pno + 1}: kotak putih tidak ditemukan, dilewati"); continue

    p.delete_image(unity["xref"]); p.delete_image(crest["xref"])
    # hanya bentuk yang SELURUHNYA di dalam kotak ini yang dibuang; lapisan gelap selebar halaman tidak ikut
    p.add_redact_annot(fitz.Rect(pill) + (-2, -2, 2, 2))
    p.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED, text=fitz.PDF_REDACT_TEXT_NONE)

    new = fitz.Rect(pill.x0, pill.y0, crest_rect.x1 + PAD_RIGHT, pill.y1)
    sh = p.new_shape(); sh.draw_rect(new, radius=0.5); sh.finish(color=None, fill=(1, 1, 1)); sh.commit()
    p.insert_image(crest_rect, pixmap=crest_pix)
    print(f"halaman {pno + 1}: logo Unity dihapus, kotak {tuple(round(v, 1) for v in pill)} -> {tuple(round(v, 1) for v in new)}")

doc.save(OUT if OUT != SRC else OUT + ".tmp", garbage=3, deflate=True)
if OUT == SRC:
    os.replace(SRC + ".tmp", SRC)
print("tersimpan:", OUT)
