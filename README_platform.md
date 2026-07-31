# KNHANES + NHANES 연구 자동화 플랫폼 (최종본)

변수·정의를 웹에서 선택하면 설계가중 분석이 내부에서 돌고, **역학·트렌드·ML 보고서가 Word로 자동 생성**됩니다.

## 구성 파일 (모두 같은 폴더)
- `factory_app.py` — Streamlit 웹앱 (진입점)
- `factory_core.py` — 로직: 로더·전처리 정의·통합변수·자동보정·설계가중 분석·Trajectory·LLM·HTML
- `engine.R` — Table 1 + 연관성(연속=선형β/이분=로지스틱OR)
- `epi_report.py` + `epi.R` + `epi_adv.R` — 역학 보고서 (Table1~6·인과추론·Love·VIF·RCS·E-value)
- `trend_report.py` + `trend.R` — 트렌드 보고서 (표준화율·Projection·APC·NB예측)
- `ml_report.py` — ML 보고서 (튜닝·전체지표·ROC/PR·threshold·calibration·SHAP·PDP)

## 설치
```bash
pip install streamlit pandas numpy pyreadstat python-docx requests matplotlib scikit-learn shap xgboost
sudo apt-get install -y r-base r-cran-survey r-cran-jsonlite r-cran-rms r-cran-mass r-cran-sandwich
ollama pull llama3.2:latest && ollama serve  # 선택. 없으면 논문 문장만 결정론 폴백
```

## 데이터
`data/KNHANES/`에 `hn08_all.sas7bdat`·`hn08_dxa.sas7bdat`(연도별), `data/NHANES/`에 `demo_j.sas7bdat` 등.
앱 사이드바에서 폴더 경로를 바꿀 수 있습니다.

## 실행
```bash
streamlit run factory_app.py
```
의료원 등 포트가 막힌 원격 서버면 SSH 터널(`ssh -L 8501:localhost:8501 ...`) 후 브라우저에서 `http://localhost:8501`.

## 탭
1. **인터랙티브 분석** — Exposure·Outcome 체크박스(모든 변수), 보정 자동, 전처리 정의 → Table 1 + 연관성 + 논문 Word
2. **Trajectory** — outcome 결정변수의 연령 궤적
3. **역학 보고서** — outcome·주요노출 → Table 1~6(SMD·crude/adj OR·하위군+P-int·IPTW/AIPW/G-comp·Love·VIF·RCS·E-value) Word
4. **트렌드 보고서** — outcome·연도 → 표준화율·Projection·APC·NB예측 Word (사이클 2개 이상)
5. **ML 보고서** — outcome·피처 → 모델 튜닝·전체지표·ROC/PR·threshold·calibration·SHAP·PDP Word

## 조작적 정의 (논문 표준)
- 당뇨: FPG≥126 or HbA1c≥6.5% or 약물 or 진단
- 고혈압: SBP≥140 or DBP≥90 or 항고혈압제
- 대사증후군: harmonized NCEP ATP III (5개 중 3개)
- 이상지질혈증: TC≥240 or HDL<40 or TG≥200
- 지방간: NAFLD-LFS>-0.640 (기본) 또는 HSI>36 / NHANES는 CAP≥248
- 흡연·음주: never vs past·current
보정변수는 노출·결과·정의성분·체격군집을 자동 제외(과보정 방지)합니다.

## 원칙
통계는 전부 R survey 설계가중(결정론). LLM(llama3.2:latest 기본값)은 변수 매핑과 보고서 문장만 작성하며 수치를 재계산하지 않습니다.
모든 소견은 hypothesis-generating이며 외부 검증이 필요합니다.
