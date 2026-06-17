import os
import re
import sys
import argparse
import difflib
import time
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

import numpy as np
import pandas as pd
import cv2
import torch
import easyocr
import streamlit as st
import requests  # 대용량 파일 분할 다운로드용 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# 웹페이지 기본 설정
st.set_page_config(page_title="스마트 의약품 안전 조회 시스템", page_icon="💊", layout="centered")

# ----------------------------------------------------
# ⚠️ [필수 확인] 허깅페이스에 올린 내 processed_db.pkl의 다운로드 주소
# ----------------------------------------------------
HUGGINGFACE_DUR_URL = "https://huggingface.co/datasets/jingjing52/dur-db/resolve/main/processed_db.pkl"

# ----------------------------------------------------
# [텍스트 및 데이터 내부 정규화 함수]
# ----------------------------------------------------
def norm_text(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"[^0-9A-Z가-힣 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1"]:
        try: return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception: continue
    raise RuntimeError(f"CSV를 읽을 수 없습니다: {path}")

# ----------------------------------------------------
# [알약 이미지 검색 엔진 구축 빌더]
# ----------------------------------------------------
@st.cache_resource
def load_pill_engines():
    full_csv = "pills.csv"
    part1 = "pills_part1.csv"
    part2 = "pills_part2.csv"
    
    if os.path.exists(full_csv):
        with st.spinner("📦 통합 알약 데이터베이스(pills.csv)를 로드하고 비전 AI를 빌드 중입니다..."):
            df = read_csv_safe(full_csv).fillna("")
    elif os.path.exists(part1) and os.path.exists(part2):
        with st.spinner("📦 분할된 알약 데이터베이스 파트를 결합하여 AI 시스템을 빌드 중입니다..."):
            df1 = read_csv_safe(part1)
            df2 = read_csv_safe(part2)
            df = pd.concat([df1, df2], ignore_index=True).fillna("")
    else:
        st.error("❌ 저장소 내부에 pills.csv 또는 분할된 csv 파일들이 존재하지 않습니다. 확인해 주세요.")
        return None, None, None, None
        
    text_cols = [c for c in ["품목명", "표시앞", "표시뒤", "표기내용앞", "표기내용뒤", "색상앞", "색상뒤", "성상", "분류명", "전문일반구분"] if c in df.columns]
    df["search_text"] = df[text_cols].astype(str).apply(
        lambda r: " ".join(norm_text(x) for x in r.values if str(x).strip() != "-"), axis=1
    )
    imprint_cols = [c for c in ["표시앞", "표시뒤", "표기내용앞", "표기내용뒤"] if c in df.columns]
    df["imprint_text"] = df[imprint_cols].astype(str).apply(
        lambda r: " ".join(norm_text(x) for x in r.values if str(x).strip() != "-"), axis=1
    ) if imprint_cols else ""
    df["imprint_text_nospace"] = df["imprint_text"].str.replace(" ", "", regex=False)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    mat = vec.fit_transform(df["search_text"].tolist())
    reader = easyocr.Reader(["en", "ko"], gpu=torch.cuda.is_available())
    return df, vec, mat, reader

df_db, tfidf_vec, tfidf_mat, ocr_reader = load_pill_engines()

# ----------------------------------------------------
# [세션 상태 설정] 데이터 유실 방지용 확실한 초기화
# ----------------------------------------------------
if "history_pills" not in st.session_state:
    st.session_state.history_pills = []
if "top_candidates_df" not in st.session_state:
    st.session_state.top_candidates_df = None
if "last_result_name" not in st.session_state:
    st.session_state.last_result_name = ""
if "last_ocr" not in st.session_state:
    st.session_state.last_ocr = ""
if "last_color" not in st.session_state:
    st.session_state.last_color = ""
if "last_shape" not in st.session_state:
    st.session_state.last_shape = ""
if "dur_danger" not in st.session_state:
    st.session_state.dur_danger = False
if "dur_msg" not in st.session_state:
    st.session_state.dur_msg = "대기 중..."

# ----------------------------------------------------
# [필규 조원님 이미지 처리 핵심 비전 함수군]
# ----------------------------------------------------
def segment_pill_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return np.ones((h, w), np.uint8) * 255

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.01 * (h * w): return np.ones((h, w), np.uint8) * 255
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask

def crop_pill_region(img: np.ndarray, padding: int = 10) -> np.ndarray:
    mask = segment_pill_mask(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img
    x, y, w, h_rect = cv2.boundingRect(cnts[0])
    H, W = img.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h_rect + padding)
    return img[y1:y2, x1:x2]

def make_ocr_variants(img: np.ndarray):
    up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    th_adapt = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return [up, cl, th_adapt]

def ocr_from_frame(reader, img: np.ndarray) -> str:
    crop = crop_pill_region(img)
    token_scores: dict = {}
    allow_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣"

    for variant in make_ocr_variants(crop):
        try:
            results = reader.readtext(variant, allowlist=allow_chars, detail=1)
            for item in results:
                text = norm_text(item[1])
                conf = float(item[2])
                if not text or conf < 0.25: continue
                for tok in text.split():
                    tok = norm_text(tok)
                    if len(tok) >= 1: token_scores[tok] = max(token_scores.get(tok, 0.0), conf)
        except Exception: pass

    if not token_scores: return ""
    ranked = sorted(token_scores.items(), key=lambda x: -x[1])
    return " ".join(tok for tok, _ in ranked[:3])

def get_color_hsv(img: np.ndarray) -> str:
    mask = segment_pill_mask(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    pixels = hsv[inner > 0]
    if len(pixels) == 0: return ""
    h, s, v = np.median(pixels, axis=0)
    if v < 50: return "검정"
    if s < 30 and v > 200: return "하양"
    if s < 30: return "회색"
    if h < 10 or h > 165: return "빨강"
    elif 10 <= h < 25: return "주황"
    elif 25 <= h < 35: return "노랑"
    elif 35 <= h < 75: return "초록"
    elif 75 <= h < 130: return "파랑"
    elif 130 <= h <= 165: return "보라"
    return "하양"

def get_shape_robust(img: np.ndarray) -> str:
    mask = segment_pill_mask(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return ""
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if area < 500 or peri == 0: return ""
    circularity = 4 * np.pi * area / (peri * peri)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    if circularity > 0.80 and aspect_ratio < 1.15: return "원형"
    elif aspect_ratio > 1.8: return "장방형"
    elif 1.15 <= aspect_ratio <= 1.8: return "타원형"
    else: return "기타"

# ----------------------------------------------------
# [DUR 크로스 대조 필터 및 매칭 검색 기법]
# ----------------------------------------------------
SIM_CHAR_MAP = str.maketrans("0QD1L852", "OOOIIBSZ")

def check_dur_danger(new_pill_name: str, pkl_db):
    if pkl_db is None or not st.session_state.history_pills:
        return False, "안전 (비교할 누적 복용 약물 없음)"

    for old_pill in st.session_state.history_pills:
        match = pkl_db[
            ((pkl_db['제품명A'].str.contains(old_pill, na=False, case=False)) & (pkl_db['제품명B'].str.contains(new_pill_name, na=False, case=False))) |
            ((pkl_db['제품명A'].str.contains(new_pill_name, na=False, case=False)) & (pkl_db['제품명B'].str.contains(old_pill, na=False, case=False)))
        ]
        if not match.empty:
            reason = match.iloc[0].get('상세정보', '병용 금기 약물 조합입니다.')
            return True, f"[{old_pill} X {new_pill_name}] 금기 사유: {reason}"
            
    return False, "국가 지정 병용금기 내역이 없습니다. (안전)"

def search_pill_from_opencv(img: np.ndarray, pkl_db):
    if df_db is None: return
    ocr_text = ocr_from_frame(ocr_reader, img)
    color    = get_color_hsv(img)
    shape    = get_shape_robust(img)

    q_vec       = tfidf_vec.transform([f"{ocr_text} {color} {shape}".strip()])
    base_scores = cosine_similarity(q_vec, tfidf_mat).ravel()

    ocr_score  = np.zeros(len(df_db), dtype=np.float32)
    ocr_joined = ocr_text.replace(" ", "")
    if ocr_joined:
        ocr_norm        = ocr_joined.translate(SIM_CHAR_MAP)
        imprint_nospace = df_db["imprint_text_nospace"].fillna("").astype(str)
        def calc_ocr_sim(db_val):
            if not db_val: return 0.0
            db_norm = db_val.translate(SIM_CHAR_MAP)
            ratio   = difflib.SequenceMatcher(None, ocr_norm, db_norm).ratio()
            if len(ocr_norm) >= 2 and (ocr_norm in db_norm or db_norm in ocr_norm): ratio = max(ratio, 0.9)
            return ratio
        ocr_score = imprint_nospace.apply(calc_ocr_sim).values.astype(np.float32)

    color_score = np.zeros(len(df_db), dtype=np.float32)
    if color:
        for c in ["색상앞", "색상뒤", "성상"]:
            if c in df_db.columns:
                color_score = np.maximum(color_score, df_db[c].astype(str).str.contains(color, na=False).astype(np.float32))

    shape_score = np.zeros(len(df_db), dtype=np.float32)
    if shape:
        for c in ["성상", "의약품제형"]:
            if c in df_db.columns:
                shape_score = np.maximum(shape_score, df_db[c].astype(str).str.contains(shape, na=False).astype(np.float32))

    final_scores = (0.50 * ocr_score + 0.20 * base_scores + 0.15 * color_score + 0.15 * shape_score)
    top_4_indices = np.argsort(-final_scores)[:4]
    
    show_cols = [c for c in ["품목명", "색상앞", "성상", "분류명"] if c in df_db.columns]
    candidates_df = df_db.iloc[top_4_indices][show_cols].copy()
    
    candidates_df.insert(0, "순위", ["🥇 1위 (매칭)", "🥈 2위", "🥉 3위", "4위"])
    candidates_df["정확도 점수"] = final_scores[top_4_indices].round(3)
    
    top_pill_name = df_db.iloc[top_4_indices[0]]["품목명"]
    danger, msg = check_dur_danger(top_pill_name, pkl_db)
    
    st.session_state.top_candidates_df = candidates_df
    st.session_state.last_result_name = top_pill_name
    st.session_state.last_ocr = ocr_text
    st.session_state.last_color = color
    st.session_state.last_shape = shape
    st.session_state.dur_danger = danger
    st.session_state.dur_msg = msg
    
    if not danger:
        if top_pill_name not in st.session_state.history_pills:
            st.session_state.history_pills.append(top_pill_name)

# 사이드바 설정
st.sidebar.title("🧭 바로가기 메뉴")
selected_page = st.sidebar.radio("이동할 페이지 선택:", ["💊 1페이지: 약물 병용금기 검색", "🍳 2페이지: AI 실시간 맞춤 레시피"])

# ----------------------------------------------------
# [DUR 데이터 로드 - requests 스트리밍 다운로드 안전 패치]
# ----------------------------------------------------
@st.cache_data
def load_dur_db():
    with st.spinner("🚀 허깅페이스 클라우드에서 국가 DUR 데이터베이스를 분할 다운로드 중..."):
        temp_pkl_path = "temp_dur_db.pkl"
        try:
            if not os.path.exists(temp_pkl_path):
                with requests.get(HUGGINGFACE_DUR_URL, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(temp_pkl_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
            return pd.read_pickle(temp_pkl_path)
        except Exception as e:
            st.error(f"❌ 허깅페이스 데이터 로드 실패: {e}")
            if os.path.exists(temp_pkl_path):
                os.remove(temp_pkl_path)
            return None

db = load_dur_db()

# ====================================================
# [PAGE 1] 약물 병용금기 검색 페이지
# ====================================================
if selected_page == "💊 1페이지: 약물 병용금기 검색":
    st.title("💊 스마트 의약품 안전 조회 시스템")
    tabs = st.tabs(["📷 브라우저 카메라 스캔", "🔍 텍스트 직접 검색"])
    
    with tabs[0]:
        if df_db is None:
            st.error("❌ 저장소 내부 알약 데이터베이스 빌드 실패 상태입니다.")
        else:
            st.subheader("📷 알약 스캔 촬영")
            st.caption("아래의 카메라 화면에 알약이 잘 보이도록 위치한 후 [Take Photo]를 눌러 촬영해 주세요.")
            
            # 💡 [서버 배포 치트키] 브라우저 내장 카메라 입력 컴포넌트 탑재
            img_file = st.camera_input("알약을 렌즈 가까이에 대고 캡처해 주세요")
            
            if img_file is not None:
                # 사용자가 사진을 찍으면 실행되는 감지 로직
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                opencv_img = cv2.imdecode(file_bytes, 1)
                
                with st.spinner("🔍 이미지 분석 및 DUR 위험도를 계산 중입니다..."):
                    search_pill_from_opencv(opencv_img, db)
            
            # 스캔 초기화 버튼 추가
            if st.button("🔄 복용 스캔 목록 초기화", use_container_width=True):
                st.session_state.history_pills = []
                st.session_state.top_candidates_df = None
                st.session_state.last_result_name = ""
                st.session_state.last_ocr = ""
                st.session_state.last_color = ""
                st.session_state.last_shape = ""
                st.session_state.dur_danger = False
                st.session_state.dur_msg = "초기화되었습니다."
                st.rerun()
            
            # 스캔 결과 출력부
            if st.session_state.last_result_name:
                st.success(f"🏆 분석 매칭 결론 1위: **{st.session_state.last_result_name}**")
                if st.session_state.top_candidates_df is not None:
                    st.dataframe(st.session_state.top_candidates_df, use_container_width=True, hide_index=True)
                if st.session_state.dur_danger: 
                    st.error(f"🚨 병용 금기: {st.session_state.dur_msg}")
                else: 
                    st.info(f"✅ 상호 복용 상태: {st.session_state.dur_msg}")
                    
            st.write("---")
            st.markdown("### 🛒 현재까지 카메라로 누적 스캔된 약물 목록")
            if st.session_state.history_pills:
                st.dataframe(pd.DataFrame({"번호": range(1, len(st.session_state.history_pills)+1), "약물 품목명": st.session_state.history_pills}), use_container_width=True, hide_index=True)
            else:
                st.caption("카메라로 사진을 촬영하여 복용할 약들을 차례대로 추가해 주세요.")

    with tabs[1]:
        st.subheader("약물 직접 확인 및 대비 검사")
        drug_A = st.text_input("첫 번째 약 이름 입력", placeholder="예: 타이레놀")
        drug_B = st.text_input("두 번째 약 이름 입력", placeholder="예: 이부프로펜")
        
        if drug_A and drug_B:
            if db is not None:
                res = db[(db['제품명A'].str.contains(drug_A, na=False, case=False)) & (db['제품명B'].str.contains(drug_B, na=False, case=False))]
                if not res.empty: 
                    st.error(f"🚨 [위험] 두 약물은 함께 복용하면 안 되는 '병용금기' 품목입니다!")
                    st.write(f"ℹ️ **금기 사유:** {res.iloc[0]['상세정보']}")
                else: 
                    st.success("✅ [안전] 국가 DUR 기준, 두 약물 간의 직접적인 병용금기 데이터가 검색되지 않았습니다.")
            else:
                st.error("❌ 허깅페이스 클라우드에서 DUR 데이터가 로드되지 않아 검색을 수행할 수 없습니다.")

# ====================================================
# [PAGE 2] AI 실시간 맞춤 레시피 추천 페이지
# ====================================================
elif selected_page == "🍳 2페이지: AI 실시간 맞춤 레시피":
    st.title("🍳 AI 실시간 의약품 정보 & 맞춤 레시피")
    gemini_key = st.sidebar.text_input("🔑 Gemini API Key 입력 (필수)", type="password")
    
    if st.session_state.history_pills:
        selected_pills = st.multiselect("대상 약물 선택:", options=st.session_state.history_pills, default=st.session_state.history_pills)
        if selected_pills and gemini_key:
            pills_string = ", ".join(selected_pills)
            prompt = f"[{pills_string}] 복용 시 피해야 할 음식과 완벽히 배제된 안전한 건강 조리법(레시피)을 상세히 알려주세요."
            with st.spinner("AI 분석 중..."):
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    st.markdown(model.generate_content(prompt).text)
                except Exception as e: 
                    st.error(f"오류 발생: {e}")
    else:
        st.warning("⚠️ 1페이지에서 카메라 스캔을 한 내역이 없습니다. 스캔을 먼저 진행해 주세요.")
