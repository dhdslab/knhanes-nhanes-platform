# -*- coding: utf-8 -*-
"""Regenerate the supplementary set for ONE dataset (arg: KNHANES or NHANES) using the full pooled
multi-cycle data. The other dataset's output is left untouched. Cleans unstable-OR pairs and rewrites
the combined summary from disk."""
import os, sys, glob, shutil, time
PROJ=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,PROJ); os.chdir(PROJ)
import factory_core as fc
import suppl_generator as sg
import pandas as pd

DS=sys.argv[1] if len(sys.argv)>1 else "KNHANES"
t0=time.time()
shutil.rmtree(os.path.join(sg.SUPPL,DS), ignore_errors=True)
for f in glob.glob(os.path.join(sg.SUPPL,f"_manifest_association_{DS}.csv")): os.remove(f)

ddir,cyc=sg.DATA[DS]
sg.log(f"[{DS}] pooled cycles {cyc}")
d=fc.apply_definitions(fc.load_raw(DS,ddir,cyc),sg.DEFS)
sg.log(f"[{DS}] pooled adults derived: {int((d.age>=20).sum())}")
a_nf,a_np,a_sk=sg.gen_association(DS,d)
t_nf,t_sk=sg.gen_trend(DS,ddir,cyc)
m_nf,m_sk=sg.gen_ml(DS,fc.load_raw(DS,ddir,cyc))

# clean unstable-OR association pairs
mp=os.path.join(sg.SUPPL,f"_manifest_association_{DS}.csv"); removed=0
if os.path.exists(mp):
    A=pd.read_csv(mp); bad=A[(A.measure=="OR")&((A.est>50)|(A.est<0.02))]
    for r in bad.itertuples():
        f=os.path.join(sg.SUPPL,DS,"association",sg.slug(r.out),f"{sg.slug(r.exp)}__{sg.slug(r.out)}_{DS}.docx")
        if os.path.exists(f): os.remove(f); removed+=1
    keep=A[~A.index.isin(bad.index)].drop(columns=["q"],errors="ignore"); keep=fc.add_fdr(keep)
    keep.to_csv(mp,index=False,encoding="utf-8-sig")
    for dgl in glob.glob(os.path.join(sg.SUPPL,DS,"association","*")):
        if os.path.isdir(dgl) and not os.listdir(dgl): os.rmdir(dgl)
sg.log(f"[{DS}] removed {removed} unstable-OR pairs")

# rebuild combined summary from disk
lines=["Supplementary summary (KNHANES 2007-2024 + NHANES 2005-2022, pooled multi-cycle)","Location: "+sg.SUPPL,""]
for ds in ["KNHANES","NHANES"]:
    a=len(glob.glob(os.path.join(sg.SUPPL,ds,"association","**","*.docx"),recursive=True))
    tr=len(glob.glob(os.path.join(sg.SUPPL,ds,"trend","*.docx")))
    ml=len(glob.glob(os.path.join(sg.SUPPL,ds,"ml","*.docx")))
    mpp=os.path.join(sg.SUPPL,f"_manifest_association_{ds}.csv"); orr=""
    if os.path.exists(mpp):
        M=pd.read_csv(mpp); o=M[M.measure=="OR"]
        orr=f", OR {o.est.min():.3g}..{o.est.max():.3g}, FDRsig {int((M.q<0.05).sum())}"
    lines+= [f"== {ds} ==",f"  association: {a} files{orr}",f"  trend: {tr} files",f"  ml: {ml} files",""]
lines+= ["KNHANES cycles: 2007-2024 (18). NHANES cycles: D-J + L, 2005-2022 (8; 2019-2020 not collected).",
         "Association: pooled (subsampled to 50k rows if larger); trend: per-cycle; ML: subsampled 6k."]
tot=len(glob.glob(os.path.join(sg.SUPPL,"**","*.docx"),recursive=True))
lines.append(f"TOTAL docx files: {tot}")
open(os.path.join(sg.SUPPL,"_SUMMARY.txt"),"w",encoding="utf-8").write("\n".join(lines)+"\n")
sg.log(f"[{DS}] regen done in {time.time()-t0:.0f}s. assoc={a_nf} trend={t_nf} ml={m_nf} total_docx={tot}")
print("DONE")
