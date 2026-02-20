import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
from gtts import gTTS
import io
import hashlib
import re

# --- 1. إعدادات الذكاء الاصطناعي ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⚠️ يرجى إضافة GEMINI_API_KEY في Secrets")
    st.stop()

def get_ai_response(prompt, image=None):
    try:
        # إضافة models/ قبل اسم الموديل لضمان التوافق
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ خطأ في الاتصال بالذكاء الاصطناعي: {str(e)}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- 2. تهيئة المجلدات وقواعد البيانات ---
for folder in ['lessons', 'exams', 'db']:
    if not os.path.exists(folder):
        os.makedirs(folder)

FILES_DB = "db/files.csv"
USERS_DB = "db/users.csv"
GRADES_DB = "db/grades.csv"

def init_db(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

init_db(USERS_DB, ["user", "pass", "role", "grade"])
init_db(FILES_DB, ["name", "grade", "sub", "type", "date"])
init_db(GRADES_DB, ["user", "sub", "score", "date"])

def load_data(path):
    try: 
        return pd.read_csv(path)
    except: 
        return pd.DataFrame()

# --- 3. الواجهة ---
st.set_page_config(page_title="منصة حسام التعليمية", layout="wide")

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

if "user_data" not in st.session_state:
    st.session_state["user_data"] = None

# --- نظام الدخول ---
if st.session_state["user_data"] is None:
    tab1, tab2 = st.tabs(["تسجيل الدخول", "حساب جديد"])
    
    with tab1:
        u = st.text_input("اسم المستخدم")
        p = st.text_input("كلمة المرور", type="password")
        if st.button("دخول"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB)
                if not users.empty:
                    match = users[(users["user"] == u) & (users["pass"] == hash_password(p))]
                    if not match.empty:
                        st.session_state["user_data"] = match.iloc[0].to_dict()
                        st.rerun()
                st.error("البيانات غير صحيحة")
                
    with tab2:
        nu = st.text_input("الاسم")
        np = st.text_input("كلمة السر الجديدة", type="password")
        nr = st.selectbox("النوع", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف", list(subs_map.keys())) if nr == "طالب" else "الكل"
        if st.button("إنشاء الحساب"):
            users = load_data(USERS_DB)
            new_u = pd.DataFrame([{"user": nu, "pass": hash_password(np), "role": nr, "grade": ng}])
            pd.concat([users, new_u], ignore_index=True).to_csv(USERS_DB, index=False)
            st.success("تم! سجل دخولك الآن")

else:
    user = st.session_state["user_data"]
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.rerun()

    # --- واجهة الأستاذ / المالك ---
    if user["role"] == "أستاذ" or user["role"] == "Owner":
        st.header("📤 مركز الرفع")
        
        # القوائم خارج الفورم لتتحدث برمجياً على الموبايل بدون مشاكل
        target_g = st.selectbox("الصف المستهدف", list(subs_map.keys()))
        target_s = st.selectbox("المادة", subs_map[target_g])
        f_type = st.radio("النوع", ["بحث", "نموذج امتحاني"])
        
        # الفورم فقط لرفع الملف
        with st.form("upload_form"):
            uploaded_file = st.file_uploader("اختر ملف PDF", type=['pdf'])
            submit = st.form_submit_button("رفع الملف الآن")
            
            if submit and uploaded_file:
                fname = f"{f_type}_{target_s}_{uploaded_file.name}".replace(" ", "_")
                path = os.path.join("lessons" if f_type == "بحث" else "exams", fname)
                
                # استخدام read() للتعامل الآمن مع الملفات المرفوعة
                with open(path, "wb") as f:
                    f.write(uploaded_file.read())
                
                f_db = load_data(FILES_DB)
                new_f = pd.DataFrame([{"name": fname, "grade": target_g, "sub": target_s, "type": f_type, "date": datetime.now().date()}])
                pd.concat([f_db, new_f], ignore_index=True).to_csv(FILES_DB, index=False)
                st.success("تم الرفع بنجاح ✅")

    # --- واجهة الطالب ---
    if user["role"] == "طالب":
        st.title(f"أهلاً {user['user']}")
        sel_sub = st.selectbox("اختر المادة", subs_map[user['grade']])
        t1, t2 = st.tabs(["📚 الدروس", "🤖 المعلم الذكي"])
        
        with t1:
            f_db = load_data(FILES_DB)
            files = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sel_sub)]
            if not files.empty:
                for _, r in files.iterrows():
                    folder = "lessons" if r['type'] == "بحث" else "exams"
                    file_path = os.path.join(folder, r['name'])
                    # التأكد من وجود الملف فعلياً قبل إظهار زر التحميل لتجنب الأخطاء
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            st.download_button(f"تحميل {r['name']}", f, file_name=r['name'])
            else: 
                st.info("لا توجد ملفات حالياً لهذه المادة.")

        with t2:
            q = st.text_input("اسأل أي سؤال...")
            if st.button("إرسال"):
                if q:
                    with st.spinner("جاري التفكير..."):
                        res = get_ai_response(f"كأستاذ، أجب الطالب في مادة {sel_sub}: {q}")
                        st.write(res)
                else:
                    st.warning("الرجاء كتابة سؤال أولاً.")
