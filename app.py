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
# 본인의 Hugging Face 아이디(유저이름)가 정확히 들어가 있는지 확인하세요.
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
    df
