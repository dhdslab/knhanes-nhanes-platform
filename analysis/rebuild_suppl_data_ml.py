# -*- coding: utf-8 -*-
"""Rebuild Supplementary Data S5 and S6 from the regenerated machine-learning reports.

The shipped S5/S6 were merged on 31 July from the reports produced under the old
model-selection rule (the winner was chosen by its held-out score). Those reports
were regenerated on 15 August after the rule was corrected to training-partition
cross-validation, but the merged bundles were never rebuilt, so the released
corpus still contradicted both the corrected code and the corrected Table 3.

Only S5 and S6 change; the association (S1, S2) and trend (S3, S4) reports were
not affected by the fix and are left untouched.
"""
import os, sys, glob, shutil, hashlib
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docxcompose.composer import Composer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
OUT = os.path.join(BASE, "Suppl")

BUNDLES = [
    ("Supplementary_Data_S5.docx", "KNHANES",
     "Supplementary Data S5. KNHANES machine-learning reports"),
    ("Supplementary_Data_S6.docx", "NHANES",
     "Supplementary Data S6. NHANES machine-learning reports"),
]

HEADNOTE = (
    "Each report is self-contained and occupies its own page. Model and "
    "hyperparameter selection use training-partition cross-validation only; the "
    "selection criterion, CV_combined, is the mean of the cross-validated area "
    "under the receiver-operating-characteristic curve and the area under the "
    "precision-recall curve. The selected model is refitted on the complete "
    "training partition and evaluated once on the untouched held-out partition, "
    "so no held-out metric enters model selection. All statistics are computed "
    "by scikit-learn; see Supplementary Appendix S7.3."
)


def title_page(title, n_reports, survey):
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = Pt(11)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title); r.bold = True; r.font.size = Pt(16)
    p2 = d.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run(f"{n_reports} machine-generated reports, one per clinical outcome, {survey}")
    r2.font.size = Pt(12)
    d.add_paragraph()
    d.add_paragraph(HEADNOTE)
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    return d


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for fname, survey, title in BUNDLES:
        src_dir = os.path.join(ROOT, "suppl", survey, "ml")
        reports = sorted(glob.glob(os.path.join(src_dir, "*.docx")))
        assert reports, f"no reports in {src_dir}"
        dst = os.path.join(OUT, fname)
        if os.path.exists(dst) and not os.path.exists(dst + ".bak_testselect"):
            shutil.copy2(dst, dst + ".bak_testselect")
            print(f"  backed up old bundle -> {os.path.basename(dst)}.bak_testselect")

        master = title_page(title, len(reports), survey)
        comp = Composer(master)
        for i, rep in enumerate(reports):
            doc = Document(rep)
            doc.paragraphs[0].insert_paragraph_before()   # keep a break between reports
            comp.append(doc)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(reports)} appended", flush=True)
        comp.save(dst)
        print(f"{fname}: {len(reports)} reports, {os.path.getsize(dst)/1e6:.1f} MB, md5 {md5(dst)[:8]}")


if __name__ == "__main__":
    main()
