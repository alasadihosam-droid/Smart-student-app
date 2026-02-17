import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai

# --- 1. إعدادات الذكاء الاصطناعي والمجلدات ---
genai.configure(api_key="AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw")
model = genai.GenerativeModel("gemini-1.5-flash")

# إنشاء نظام الملفات
base_dir = "platform_data"
folders = ['lessons', 'exams', 'keys', 'db']
for f in folders:
    os.makedirs(os.path.join(base_dir, f), exist_ok=True)

USERS_DB = os.path.join(base_dir, "db/users.csv")
FILES_DB = os.path.join(base_dir, "db/files.csv")

def load_db(path, columns):
    if os.path.exists(path): return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# --- 2. واجهة المستخدم والتصميم ---
st.set_page_config(page_title="منصة الطالب الذكي", layout="wide")

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 3. نظام الدخول الآمن ---
if "auth" not in st.session_state:
    st.sidebar.title("🔐 تسجيل الدخول")
    mode = st.sidebar.radio("الوضع:", ["دخول", "إنشاء حساب"])
    u_in = st.sidebar.text_input("اسم المستخدم")
    p_in = st.sidebar.text_input("كلمة المرور", type="password")

    if mode == "دخول":
        if st.sidebar.button("تسجيل دخول"):
            # دخول المدير (حسام)
            if u_in == "Hosam" and p_in == "Anahosam031007":
                st.session_state["auth"] = {"user": u_in, "role": "Owner"}
                st.rerun()
            else:
                users = load_db(USERS_DB, ["user", "pass", "role", "grade"])
                match = users[(users["user"] == u_in) & (users["pass"] == p_in)]
                if not match.empty:
                    st.session_state["auth"] = {"user": u_in, "role": match.iloc[0]["role"], "grade": match.iloc[0]["grade"]}
                    st.rerun()
                else: st.error("بيانات خاطئة")
    else:
        role_in = st.sidebar.selectbox("الرتبة:", ["طالب", "أستاذ"])
        grade_in = st.sidebar.selectbox("الصف:", list(subs_map.keys())) if role_in == "طالب" else "الكل"
        if st.sidebar.button("إنشاء الحساب"):
            users = load_db(USERS_DB, ["user", "pass", "role", "grade"])
            if u_in in users["user"].values: st.error("المستخدم موجود!")
            else:
                new_u = pd.DataFrame([{"user": u_in, "pass": p_in, "role": role_in, "grade": grade_in}])
                pd.concat([users, new_u]).to_csv(USERS_DB, index=False)
                st.success("✅ تم بنجاح! سجل دخولك الآن.")

else:
    auth = st.session_state["auth"]
    st.sidebar.success(f"مرحباً: {auth['user']}")
    if st.sidebar.button("تسجيل الخروج"):
        del st.session_state["auth"]; st.rerun()

    # --- 4. واجهة المدير (حسام) ---
    if auth["role"] == "Owner":
        st.title("👑 لوحة التحكم المطلقة")
        t1, t2 = st.tabs(["👥 الأعضاء", "📂 المنشورات"])
        with t1:
            users = load_db(USERS_DB, ["user", "pass", "role", "grade"])
            for i, row in users.iterrows():
                c1, c2 = st.columns([3, 1])
                c1.write(f"👤 {row['user']} ({row['role']})")
                if c2.button("حذف", key=f"u_{i}"):
                    users.drop(i).to_csv(USERS_DB, index=False); st.rerun()
        with t2:
            files = load_db(FILES_DB, ["name", "grade", "sub", "type"])
            for i, row in files.iterrows():
                c1, c2 = st.columns([3, 1])
                c1.write(f"📄 {row['name']} | {row['grade']} - {row['sub']}")
                if c2.button("حذف", key=f"f_{i}"):
                    files.drop(i).to_csv(FILES_DB, index=False); st.rerun()

    # --- 5. واجهة الأستاذ ---
    elif auth["role"] == "أستاذ":
        st.title("👨‍🏫 لوحة نشر المحتوى")
        col1, col2 = st.columns(2)
        with col1: target_g = st.selectbox("الصف:", list(subs_map.keys()))
        with col2: target_s = st.selectbox("المادة:", subs_map[target_g])
        f_type = st.radio("نوع الملف:", ["بحث PDF", "نموذج امتحان", "سلم تصحيح"])
        up = st.file_uploader(f"ارفع {f_type}")
        if up and st.button("نشر الآن"):
            files = load_db(FILES_DB, ["name", "grade", "sub", "type"])
            new_f = pd.DataFrame([{"name": up.name, "grade": target_g, "sub": target_s, "type": f_type}])
            pd.concat([files, new_f]).to_csv(FILES_DB, index=False)
            folder = {"بحث PDF":"lessons", "نموذج امتحان":"exams", "سلم تصحيح":"keys"}[f_type]
            with open(os.path.join(base_dir, folder, up.name), "wb") as f: f.write(up.getbuffer())
            st.success("✅ تم النشر!")

    # --- 6. واجهة الطالب ---
    else:
        st.title(f"🎓 بوابة الطالب: {auth['grade']}")
        sub = st.selectbox("اختر المادة:", subs_map[auth["grade"]])
        tab_f, tab_ai = st.tabs(["📚 ملفات المادة", "🤖 المساعد الذكي"])
        
        with tab_f:
            files = load_db(FILES_DB, ["name", "grade", "sub", "type"])
            my_f = files[(files["grade"] == auth["grade"]) & (files["sub"] == sub)]
            if not my_f.empty:
                for _, r in my_f.iterrows():
                    folder = {"بحث PDF":"lessons", "نموذج امتحان":"exams", "سلم تصحيح":"keys"}[r['type']]
                    with open(os.path.join(base_dir, folder, r['name']), "rb") as f:
                        st.download_button(f"📥 تحميل {r['type']}: {r['name']}", f, file_name=r['name'])
            else: st.info("لا ملفات مرفوعة.")

        with tab_ai:
            if st.button("📝 توليد أسئلة امتحان ذكية"):
                with st.spinner("جاري التوليد..."):
                    res = model.generate_content(f"اكتب نموذج امتحاني شامل لمادة {sub} لصف {auth['grade']} حسب المنهج السوري.")
                    st.write(res.text)
            st.divider()
            img = st.file_uploader("📸 ارفع صورة حلك للتصحيح", type=["jpg", "png", "jpeg"])
            if img and st.button("بدأ التصحيح"):
                with st.spinner("الذكاء الاصطناعي يحلل ورقتك..."):
                    res = model.generate_content([f"صحح هذا الحل لمادة {sub} صف {auth['grade']} وأعطِ درجة من 100 مع شرح الأخطاء.", Image.open(img)])
                    st.markdown("### 📝 نتيجة التصحيح:")
                    st.write(res.text)
