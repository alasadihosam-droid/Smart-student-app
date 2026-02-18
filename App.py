import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai

# --- 1. إعدادات الذكاء الاصطناعي ---
genai.configure(api_key="AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw")
model = genai.GenerativeModel("gemini-1.5-flash")

# --- 2. نظام المجلدات ---
for folder in ['lessons', 'exams', 'keys', 'db']:
    os.makedirs(folder, exist_ok=True)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"

@st.cache_data(ttl=5)
def load_data(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# --- 3. تصميم الواجهة ---
st.set_page_config(page_title="منصة الطالب الذكي", layout="wide")

st.markdown("""
    <style>
    .main .block-container { max-width: 900px; padding-bottom: 10rem; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #D32F2F; color: white; font-weight: bold; border: none; }
    .upload-box { border: 1px dashed #D32F2F; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 4. الدخول ---
if "user_data" not in st.session_state:
    st.title("🚀 منصة الطالب الذكي")
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
            st.success("تم بنجاح")

else:
    user = st.session_state["user_data"]
    st.sidebar.title(f"👋 {user['name']}")
    if st.sidebar.button("خروج"):
        del st.session_state["user_data"]; st.rerun()

    # --- واجهة الأستاذ (المطورة) ---
    if user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع الملفات")
        col_g, col_s = st.columns(2)
        with col_g: tg = st.selectbox("اختر الصف:", list(subs_map.keys()))
        with col_s: ts = st.selectbox("اختر المادة:", subs_map[tg])
        
        st.divider()
        
        # خانات الرفع المنفصلة
        def upload_func(label, folder, type_name):
            st.markdown(f"**📍 {label}**")
            up = st.file_uploader(f"ارفع {label}", key=type_name)
            if up and st.button(f"تأكيد رفع {label}", key=f"btn_{type_name}"):
                f_name = f"{type_name}_{ts}_{up.name}"
                with open(os.path.join(folder, f_name), "wb") as f: f.write(up.getbuffer())
                f_db = load_data(FILES_DB, ["name", "grade", "sub", "type"])
                pd.concat([f_db, pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_name}])]).to_csv(FILES_DB, index=False)
                st.success(f"تم رفع {label} بنجاح!")

        c1, c2, c3 = st.columns(3)
        with c1: upload_func("ملف البحث (PDF)", "lessons", "بحث")
        with c2: upload_func("نموذج الامتحان", "exams", "نموذج")
        with c3: upload_func("سلم التصحيح", "keys", "سلم")

    # --- واجهة الطالب (المطورة) ---
    elif user["role"] == "طالب":
        st.header(f"🎓 بوابة {user['grade']}")
        sub = st.selectbox("اختر المادة:", subs_map[user['grade']])
        t_study, t_ai = st.tabs(["📚 ملفات الأستاذ", "🤖 المساعد الذكي"])
        
        with t_study:
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type"])
            my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
            if not my_f.empty:
                for _, r in my_f.iterrows():
                    folder = {"بحث": "lessons", "نموذج": "exams", "سلم": "keys"}[r['type']]
                    file_path = os.path.join(folder, r['name'])
                    with open(file_path, "rb") as f:
                        # إضافة application/pdf لضمان التحميل بشكل صحيح
                        st.download_button(f"📥 تحميل {r['type']}: {r['name']}", f, file_name=r['name'], mime="application/pdf")
            else: st.info("لا يوجد ملفات حالياً")

        with t_ai:
            if st.button("📝 توليد نموذج امتحاني شامل"):
                with st.spinner("الذكاء الاصطناعي يقوم بصياغة الأسئلة..."):
                    # أمر محسن للذكاء الاصطناعي
                    prompt = f"أنت أستاذ سوري خبير. اكتب نموذج امتحان لمادة {sub} لصف {user['grade']} وفق منهاج وزارة التربية السورية. اجعل الأسئلة متنوعة (اختيار من متعدد، تعاريف، مسائل) مع توزيع الدرجات."
                    res = model.generate_content(prompt)
                    st.markdown(res.text)
            
            st.divider()
            img = st.file_uploader("📸 تصحيح حل الطالب (ارفع صورة)", type=["jpg", "png", "jpeg"])
            if img and st.button("✨ ابدأ التصحيح الفوري"):
                with st.spinner("جاري التحليل..."):
                    res = model.generate_content([f"صحح هذه الورقة لمادة {sub} {user['grade']} منهاج سوري، أعطِ ملاحظات دقيقة وعلامة من 100.", Image.open(img)])
                    st.success("تم التصحيح!")
                    st.write(res.text)

    # --- واجهة المدير (حسام) ---
    elif user["role"] == "Owner":
        st.header("👑 لوحة الملك حسام")
        u_df = load_data(USERS_DB, ["user", "pass", "role", "grade"])
        st.dataframe(u_df, use_container_width=True)
        if st.button("حذف كل البيانات (للتنظيف)"):
            if os.path.exists(FILES_DB): os.remove(FILES_DB)
            st.rerun()
