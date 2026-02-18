import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime

# --- 1. إعدادات الذكاء الاصطناعي ---
genai.configure(api_key="AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw")
model = genai.GenerativeModel("gemini-1.5-flash")

# --- 2. نظام المجلدات وقواعد البيانات ---
for folder in ['lessons', 'exams', 'keys', 'db']:
    os.makedirs(folder, exist_ok=True)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
GRADES_DB = "db/grades.csv"

def load_data(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# --- 3. تحديد الوقت والثيم التلقائي ---
now = datetime.now()
hour = now.hour

# تحديد عبارة الترحيب
if 5 <= hour < 12:
    greeting = "☀️ صباح الخير"
    theme_mode = "light"
elif 12 <= hour < 17:
    greeting = "🌤️ طاب يومك"
    theme_mode = "light"
elif 17 <= hour < 21:
    greeting = "🌆 مساء الخير"
    theme_mode = "dark"
else:
    greeting = "🌙 ليلة سعيدة"
    theme_mode = "dark"

# --- 4. تصميم الواجهة (تعديل الألوان بناءً على الوقت) ---
st.set_page_config(page_title="منصة الطالب الذكي", layout="wide")

if theme_mode == "dark":
    bg_color = "#121212"
    text_color = "#FFFFFF"
    card_bg = "#1E1E1E"
else:
    bg_color = "#F5F7F9"
    text_color = "#000000"
    card_bg = "#FFFFFF"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .main .block-container {{ max-width: 900px; padding-bottom: 10rem; }}
    .stButton>button {{ 
        width: 100%; border-radius: 10px; height: 3.5em;
        background-color: #D32F2F; color: white; font-weight: bold; border: none;
    }}
    .greeting-box {{
        padding: 20px; background-color: {card_bg}; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
        text-align: center; border: 1px solid #D32F2F;
    }}
    .notif {{ padding: 10px; background-color: #FFF3E0; border-right: 5px solid #FF9800; border-radius: 5px; margin-bottom: 10px; color: #333; }}
    </style>
    """, unsafe_allow_html=True)

# خريطة المواد
subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 5. الدخول ---
if "user_data" not in st.session_state:
    st.markdown(f'<div class="greeting-box"><h1>{greeting}</h1><p>مرحباً بك في منصة الأستاذ حسام</p></div>', unsafe_allow_html=True)
    t_log, t_sign = st.tabs(["🔐 دخول", "📝 حساب جديد"])
    with t_log:
        u, p = st.text_input("اسم المستخدم"), st.text_input("كلمة المرور", type="password")
        if st.button("تسجيل الدخول"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"name": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
                match = users[(users["user"] == u) & (users["pass"] == p)]
                if not match.empty:
                    st.session_state["user_data"] = {"name": u, "role": match.iloc[0]["role"], "grade": match.iloc[0]["grade"]}
                    st.rerun()
                else: st.error("خطأ بالبيانات")
    with t_sign:
        nu, np = st.text_input("الاسم الجديد"), st.text_input("كلمة المرور الجديدة", type="password")
        nr = st.selectbox("الرتبة", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف", list(subs_map.keys())) if nr == "طالب" else "الكل"
        if st.button("إنشاء الحساب"):
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            pd.concat([users, pd.DataFrame([{"user": nu, "pass": np, "role": nr, "grade": ng}])]).to_csv(USERS_DB, index=False)
            st.success("تم بنجاح، سجل دخولك الآن")

else:
    user = st.session_state["user_data"]
    st.sidebar.title(f"👋 {user['name']}")
    st.sidebar.info(f"{greeting}")
    if st.sidebar.button("خروج"):
        del st.session_state["user_data"]; st.rerun()

    # --- واجهة المدير (حسام) ---
    if user["role"] == "Owner":
        st.header(f"👑 لوحة الملك حسام - {greeting}")
        t_u, t_g, t_f = st.tabs(["👤 الأعضاء", "📊 سجل العلامات", "📂 الملفات"])
        with t_g: st.dataframe(load_data(GRADES_DB, ["student", "subject", "grade", "date"]), use_container_width=True)
        with t_u: st.dataframe(load_data(USERS_DB, ["user", "pass", "role", "grade"]), use_container_width=True)
        with t_f: st.dataframe(load_data(FILES_DB, ["name", "grade", "sub", "type", "date"]), use_container_width=True)

    # --- واجهة الأستاذ ---
    elif user["role"] == "أستاذ":
        st.header(f"👨‍🏫 مركز الرفع - {greeting}")
        tg = st.selectbox("الصف المستهدف:", list(subs_map.keys()))
        ts = st.selectbox("المادة:", subs_map[tg])
        
        def upload_func(label, folder, type_name):
            up = st.file_uploader(f"اختر ملف {label}", key=f"up_{type_name}_{tg}_{ts}")
            if up and st.button(f"تأكيد رفع {label}", key=f"btn_{type_name}_{tg}_{ts}"):
                safe_name = up.name.replace(" ", "_")
                f_name = f"{type_name}_{ts}_{safe_name}"
                with open(os.path.join(folder, f_name), "wb") as f: f.write(up.getbuffer())
                f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
                pd.concat([f_db, pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_name, "date": datetime.now().strftime("%Y-%m-%d")}])]).to_csv(FILES_DB, index=False)
                st.success(f"✅ تم رفع {label} بنجاح!")
                st.cache_data.clear()

        c1, c2, c3 = st.columns(3)
        with c1: upload_func("البحث", "lessons", "بحث")
        with c2: upload_func("النموذج", "exams", "نموذج")
        with c3: upload_func("السلم", "keys", "سلم")

    # --- واجهة الطالب ---
    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h2>{greeting} يا بطل!</h2><p>جاهز لدرس اليوم؟</p></div>', unsafe_allow_html=True)
        
        f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
        today = datetime.now().strftime("%Y-%m-%d")
        new_files = f_db[(f_db["grade"] == user["grade"]) & (f_db["date"] == today)]
        if not new_files.empty:
            for _, r in new_files.iterrows():
                st.markdown(f'<div class="notif">🔔 إشعار: أضاف الأستاذ {r["type"]} جديد في مادة {r["sub"]}</div>', unsafe_allow_html=True)

        sub = st.selectbox("اختر المادة:", subs_map[user['grade']])
        t_study, t_ai, t_chat = st.tabs(["📚 المكتبة", "🤖 المصحح", "💬 اسأل المنهاج"])
        
        with t_study:
            my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
            if not my_f.empty:
                for _, r in my_f.iterrows():
                    folder = {"بحث": "lessons", "نموذج": "exams", "سلم": "keys"}[r['type']]
                    path = os.path.join(folder, r['name'])
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(f"📥 تحميل {r['type']}: {r['name'].split('_')[-1]}", f, file_name=r['name'], mime="application/pdf")
            else: st.info("لا توجد ملفات مرفوعة حالياً.")

        with t_ai:
            img = st.file_uploader("📸 ارفع صورة حلك", type=["jpg", "png", "jpeg"])
            if img and st.button("✨ تصحيح وحفظ العلامة"):
                with st.spinner("جاري التصحيح..."):
                    res = model.generate_content([f"أستاذ سوري، صحح ورقة {sub} {user['grade']} منهاج سوري. أعطِ علامة من 100.", Image.open(img)])
                    st.write(res.text)
                    g_db = load_data(GRADES_DB, ["student", "subject", "grade", "date"])
                    pd.concat([g_db, pd.DataFrame([{"student": user['name'], "subject": sub, "grade": res.text[:30], "date": today}])]).to_csv(GRADES_DB, index=False)

        with t_chat:
            user_q = st.chat_input("اسألني أي شيء في المنهاج...")
            if user_q:
                res = model.generate_content(f"أنت مساعد تعليمي للمنهاج السوري لصف {user['grade']}. أجب عن: {user_q}")
                st.chat_message("assistant").write(res.text)
