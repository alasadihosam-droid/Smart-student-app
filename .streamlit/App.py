import streamlit as st
import pandas as pd
from PIL import Image
import datetime
import hashlib
import google.generativeai as genai
import os

# إعدادات المسارات
base_path = "data"
if not os.path.exists(base_path): os.makedirs(base_path)
upload_path = os.path.join(base_path, 'uploads')
if not os.path.exists(upload_path): os.makedirs(upload_path)

# إعداد الذكاء الاصطناعي (مفتاحك)
genai.configure(api_key="AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw")
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="منصة الطالب الذكي", layout="wide")

# تصميم الواجهة
st.markdown('''<style>
    .stApp { background-color: white; }
    h1 { color: #D32F2F; text-align:center; border-bottom: 2px solid #1E1E1E; padding-bottom:10px; }
    .stButton>button { background-color:#D32F2F; color:white; border-radius:8px; font-weight:bold; }
    [data-testid="stSidebar"] { background-color:#1E1E1E; color:white; }
</style>''', unsafe_allow_html=True)

st.title("🚀 منصة الطالب الذكي")

CSV_DB = os.path.join(base_path, "results.csv")
USERS_DB = os.path.join(base_path, "users.csv")

if os.path.exists(USERS_DB): users = pd.read_csv(USERS_DB)
else: users = pd.DataFrame(columns=["username", "password", "role", "grade"])

if os.path.exists(CSV_DB): results = pd.read_csv(CSV_DB)
else: results = pd.DataFrame(columns=["الاسم", "الصف", "المادة", "العلامة", "التاريخ"])

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "عربي"]
}

def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()

with st.sidebar:
    st.header("🔐 الدخول")
    if "logged_in" not in st.session_state:
        auth_mode = st.radio("اختر:", ["تسجيل دخول", "إنشاء حساب"])
        u_in = st.text_input("اسم المستخدم")
        p_in = st.text_input("كلمة المرور", type="password")

        if auth_mode == "إنشاء حساب":
            role_in = st.selectbox("النوع:", ["🎓 طالب", "👨‍🏫 أستاذ"])
            grade_in = st.selectbox("الصف:", list(subs_map.keys())) if role_in == "🎓 طالب" else "None"
            if st.button("تأكيد التسجيل"):
                if u_in and p_in:
                    new_u = pd.DataFrame([{"username": u_in, "password": hash_password(p_in), "role": role_in, "grade": grade_in}])
                    users = pd.concat([users, new_u], ignore_index=True)
                    users.to_csv(USERS_DB, index=False)
                    st.success("✅ تم الإنشاء")
        else:
            if st.button("دخول"):
                match = users[(users["username"].astype(str) == u_in) & (users["password"].astype(str) == hash_password(p_in))]
                if not match.empty:
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u_in
                    st.session_state["role"] = match.iloc[0]["role"]
                    st.rerun()
                else: st.error("خطأ في البيانات")
    else:
        st.write(f"مرحباً: {st.session_state['user']}")
        if st.button("خروج"):
            del st.session_state["logged_in"]
            st.rerun()

if "logged_in" in st.session_state:
    role = st.session_state["role"]
    username = st.session_state["user"]

    if role == "👨‍🏫 أستاذ":
        st.subheader(f"لوحة التحكم: {username}")
        up = st.file_uploader("ارفع ملف المادة", type=["pdf", "jpg", "png"])
        if up and st.button("نشر"):
            with open(os.path.join(upload_path, up.name), "wb") as f:
                f.write(up.getbuffer())
            st.success("تم النشر بنجاح!")
    else:
        st.subheader(f"بوابة الطالب: {username}")
        # تبويبات الطالب
        t1, t2 = st.tabs(["📚 الملفات", "📸 التصحيح"])
        with t1:
            st.write("الملفات المرفوعة ستظهر هنا")
        with t2:
            img = st.file_uploader("ارفع الحل")
            if img and st.button("تصحيح"):
                res = model.generate_content(["صحح العلامة من 10:", Image.open(img)])
                st.write(res.text)
