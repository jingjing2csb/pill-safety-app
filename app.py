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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import gdown

# 웹페이지 기본 설정
st.set_page_config(page_title="스마트 의약품 안전 조회 시스템", page_icon="💊", layout="centered")

DEFAULT_CSV_PATH = "pills.csv"
DEFAULT_PKL_PATH = "processed_db.pkl"

# ----------------------------------------------------
# [구글 드라이브 대용량 파일 자동 다운로드 로직]
# ----------------------------------------------------
@st.cache_resource
def download_large_db_files():
    # 1. pills.csv 다운로드 (약 40MB 내외 대용량 대비)
    if not os.path.exists(DEFAULT_CSV_PATH):
        with st.spinner("📦 초기에 필요한 알약 데이터베이스(pills.csv)를 안전하게 다운로드 중입니다... (약 10~20초 소요)"):
            csv_id = "1fM2H1Xdp6w-07TE3FcPfV3d2_lcyK-DH"
            csv_url = f"https://drive.google.com/uc?id={csv_id}"
            gdown.download(csv_url, DEFAULT_CSV_PATH, quiet=True)
            
    # 2. processed_db.pkl 다운로드
    if not os.path.exists(DEFAULT_PKL_PATH):
        with st.spinner("🛡️ 국가 병용금기 데이터베이스(processed_db.pkl)를 안전하게 다운로드 중입니다..."):
            pkl_id = "1i26f4a9f5P-HI6yFZMTvaxSpoPoy8WLw"
            pkl_url = f"https://drive.google.com/uc?id={pkl_id}"
            gdown.download(pkl_url, DEFAULT_PKL_PATH, quiet=True)

# 인터넷 상에 배포되었을 때 실행할 파일 다운로드 구동
download_large_db_files()


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
# [음식 매칭 DB] 금기 음식과 사유 정보
# ----------------------------------------------------
DRUG_FOOD_RULES = {
    "아세트아미노펜": {
        "aliases": ["타이레놀", "펜잘", "게보린", "아세트아미노펜"],
        "bad_foods": "술(알코올), 과도한 카페인(커피, 에너지드링크)",
        "reason": "알코올과 함께 섭취 시 간 손상 위험이 극도로 높아지며, 카페인은 약물 부작용을 증가시킵니다."
    },
    "이부프로펜": {
        "aliases": ["이부프로펜", "덱시부프로펜", "아스피린", "애드빌", "이지엔"],
        "bad_foods": "귤, 오렌지 같은 산성 과일, 탄산음료, 술(알코올)",
        "reason": "위점막을 자극하는 약물이므로 산성 음식과 만나면 속 쓰림, 위염을 유발할 수 있습니다."
    },
    "고지혈증약": {
        "aliases": ["고지혈증약", "리피토", "아토르바스타틴", "로수바스타틴"],
        "bad_foods": "자몽, 자몽주스, RED와인",
        "reason": "자몽 속 성분이 약물의 체내 분해를 방해하여 혈중 농도를 높여 근육통 등 부작용을 유발합니다."
    }
}

# ----------------------------------------------------
# [필규 조원님 비전/AI 알고리즘 캐싱 로드]
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

@st.cache_resource
def load_pill_engines():
    if not os.path.exists(DEFAULT_CSV_PATH):
        return None, None, None, None
    df = read_csv_safe(DEFAULT_CSV_PATH).fillna("")
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

# ----------------------------------------------------
# [카메라 창 팝업 호출 실행기]
# ----------------------------------------------------
def open_cv_camera_popup(pkl_db):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        H, W = frame.shape[:2]
        cx, cy = W // 2, H // 2
        size = min(W, H) // 4
        
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)
        cv2.putText(frame, "PLACE PILL HERE & PRESS SPACE TO CAPTURE", (cx - size, cy - size - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Exit: Q or ESC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Pill Scanner (Press SPACE)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            roi = frame[cy - size:cy + size, cx - size:cx + size]
            search_pill_from_opencv(roi, pkl_db)
            break
            
    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------
# 사이드바 설정 (페이지 전환)
# ----------------------------------------------------
st.sidebar.title("🧭 바로가기 메뉴")
selected_page = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    ["💊 1페이지: 약물 병용금기 검색", "🍳 2페이지: AI 실시간 맞춤 레시피"]
)

# [데이터 로드 함수]
@st.cache_data
def load_dur_db():
    try:
        return pd.read_pickle(DEFAULT_PKL_PATH)
    except FileNotFoundError:
        return None

db = load_dur_db()


# ====================================================
# [PAGE 1] 약물 병용금기 검색 페이지
# ====================================================
if selected_page == "💊 1페이지: 약물 병용금기 검색":
    st.title("💊 스마트 의약품 안전 조회 시스템")
    st.write("인식된 약물 이름을 기반으로 약과 약 사이의 안전성을 검사합니다.")
    
    tabs = st.tabs(["📷 카메라 스캔 (AR 가상 영역)", "🔍 텍스트 직접 검색"])
    
    with tabs[0]:
        st.subheader("바닥에 있는 약 스캔하기")
        
        if df_db is None:
            st.error("❌ 데이터베이스 로드 실패 상태입니다.")
        else:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📷 실시간 카메라 판독 시작", type="primary", use_container_width=True):
                    open_cv_camera_popup(db)
                    st.rerun()
            with col_btn2:
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
            
            if st.session_state.last_result_name:
                st.write("---")
                st.success(f"🏆 분석 매칭 결론 1위: **{st.session_state.last_result_name}**")
                
                if st.session_state.top_candidates_df is not None:
                    st.markdown("##### 🔍 컴퓨터 비전 매칭 상위 4개 후보군 목록")
                    st.dataframe(st.session_state.top_candidates_df, use_container_width=True, hide_index=True)
                
                st.text(f"💡 [추출 정보] OCR 글자: {st.session_state.last_ocr or '-'} | 색상: {st.session_state.last_color or '-'} | 형태: {st.session_state.last_shape or '-'}")
                
                if st.session_state.dur_danger:
                    st.error(f"🚨 **병용 금기 판정:** {st.session_state.dur_msg}")
                else:
                    st.info(f"✅ **상호 복용 상태:** {st.session_state.dur_msg}")
            
            st.write("---")
            st.markdown("### 🛒 현재까지 카메라로 누적 스캔된 약물 목록")
            if st.session_state.history_pills:
                st.dataframe(pd.DataFrame({"번호": range(1, len(st.session_state.history_pills)+1), "약물 품목명": st.session_state.history_pills}), use_container_width=True, hide_index=True)
            else:
                st.caption("카메라 판독 단추를 눌러 복용할 약들을 차례대로 추가해 주세요.")
        
    with tabs[1]:
        st.subheader("약물 직접 확인 및 대비 검사")
        if db is None:
            st.error("❌ 전처리된 데이터 파일 로드 실패")
        else:
            col1, col2 = st.columns(2)
            with col1:
                drug_A = st.text_input("첫 번째 약 이름 입력", placeholder="예: 타이레놀", key="dur_a")
            with col2:
                drug_B = st.text_input("두 번째 약 이름 입력", placeholder="예: 이부프로펜", key="dur_b")
                
            if drug_A and drug_B:
                result = db[(db['제품명A'].str.contains(drug_A, na=False, case=False)) & (db['제품명B'].str.contains(drug_B, na=False, case=False))]
                if not result.empty:
                    st.error("🚨 [위험] 두 약물은 함께 복용하면 안 되는 '병용금기' 품목입니다!")
                    st.write(f"ℹ️ **금기 사유:** {result.iloc[0]['상세정보']}")
                else:
                    st.success("✅ [안전] 국가 DUR 기준, 두 약물 간의 직접적인 병용금기 데이터가 검색되지 않았습니다.")


# ====================================================
# [PAGE 2] AI 실시간 맞춤 레시피 추천 페이지
# ====================================================
elif selected_page == "🍳 2페이지: AI 실시간 맞춤 레시피":
    st.title("🍳 AI 실시간 의약품 정보 & 맞춤 레시피")
    st.write("카메라로 스캔한 약들을 선택하여 피해야 할 음식과 안전한 건강 레시피를 한 번에 조회합니다.")
    
    gemini_key = st.sidebar.text_input("🔑 Gemini API Key 입력 (필수)", type="password")
    
    st.subheader("📋 분석할 약물 선택")
    
    if st.session_state.history_pills:
        selected_pills = st.multiselect(
            "레시피를 조회할 약물을 선택하세요 (복수 선택 가능):",
            options=st.session_state.history_pills,
            default=st.session_state.history_pills
        )
    else:
        st.warning("⚠️ 1페이지에서 카메라 스캔을 한 내역이 없습니다. 아래 직접 입력을 이용하거나 먼저 스캔해 주세요.")
        selected_pills = []
        
    manual_query = st.text_input("목록에 없는 약물을 추가로 입력하여 조회할 수도 있습니다. (선택사항)", placeholder="예: 아스피린, 당뇨약 등")
    
    final_search_list = list(selected_pills)
    if manual_query.strip():
        final_search_list.append(manual_query.strip())
        
    if final_search_list:
        pills_string = ", ".join(final_search_list)
        
        if not gemini_key:
            st.warning("⚠️ 실시간 약물 정보 조회를 위해 왼쪽 사이드바에 Gemini API Key를 반드시 입력해 주세요!")
        else:
            st.subheader(f"🔍 AI가 분석한 [{pills_string}] 종합 정보")
            
            prompt = f"""
            사용자가 현재 다음 약물들을 복용 중이거나 이에 대해 알고 싶어 합니다: [{pills_string}]
            대형 언어 모델로서 가진 정확한 의학/약학 지식을 바탕으로 아래 양식에 맞추어 답변해 주세요.
            제시된 약물들이 여러 개라면, 그 약물들이 공통으로 피해야 하거나 각각 피해야 하는 음식들을 모두 종합해서 알려주세요.

            [출력 양식]
            ## ❌ 복용 시 피해야 할 음식 (병용 금기 음식)
            (이 약물들과 함께 먹으면 절대 안 되는 음식이나 음료 목록을 나열해 주세요.)

            ## ⚠️ 위험 이유
            (왜 해당 음식들을 피해야 하는지 구체적인 부작용이나 기전을 초보자도 이해하기 쉽게 설명해 주세요. 어떤 약물 때문에 발생하는지도 명시해 주세요.)

            ---
            ## 👨‍🍳 AI 추천 안전 건강 조리법
            * **요리 이름:** (위에서 언급한 모든 금기 음식을 '완벽히 배제'하고, 환자의 건강과 약효에 도움이 되는 식재료로 구성된 요리 이름)
            * **📋 준비할 재료:** (재료 목록)
            * **🍳 조리 순서:** (1, 2, 3 단계별 설명)
            """
            
            with st.spinner("AI 비서가 의학 데이터베이스를 탐색하여 통합 레시피를 생성 중입니다..."):
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash') 
                    response = model.generate_content(prompt)
                    
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"AI 호출 중 오류가 발생했습니다: {e}")
    else:
        st.info("조회할 약물을 상단에서 선택하거나 이름을 입력해 주세요.")