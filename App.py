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
import requests

# ==========================================
# 1. إعدادات الأمان والذكاء الاصطناعي
# ==========================================
try:
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("⚠️ مفتاح API غير موجود. يرجى إضافة GEMINI_API_KEY في ملف Secrets.")
        st.stop()
        
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
except Exception as e:
    st.error(f"⚠️ خطأ في الوصول إلى Secrets: {e}")
    st.stop()

genai.configure(api_key=API_KEY)

def get_ai_response(prompt, image=None):
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        safe_models = [m for m in available_models if "2.5" not in m]
        if not safe_models: return "⚠️ عذراً، جميع الموديلات المتاحة في حسابك غير مجانية."
        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                if image: return model.generate_content([prompt, image]).text
                else: return model.generate_content(prompt).text
            except Exception: continue 
        return "⚠️ تم رفض الاتصال. جرب تشغيل VPN."
    except Exception as e: return f"⚠️ خطأ عام: {str(e)}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except: return None

def get_telegram_updates(token):
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates").json()
        if r.get("ok"): return r["result"]
    except: pass
    return []

def download_telegram_file(token, file_id, dest_path):
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}").json()
        if r.get("ok"):
            file_path = r["result"]["file_path"]
            file_data = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}").content
            with open(dest_path, "wb") as f: f.write(file_data)
            return True
    except: pass
    return False

# ==========================================
# 2. تهيئة قواعد البيانات والمجلدات
# ==========================================
for folder in ['lessons', 'exams', 'db', 'profiles']:
    if not os.path.exists(folder): os.makedirs(folder)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
GRADES_DB = "db/grades.csv"
NOTIFY_DB = "db/notifications.csv" # قاعدة بيانات تنويهات الأساتذة

def init_db(path, columns):
    if not os.path.exists(path): pd.DataFrame(columns=columns).to_csv(path, index=False)

init_db(USERS_DB, ["user", "pass", "role", "grade", "fb_link"])
init_db(FILES_DB, ["name", "grade", "sub", "type", "date"])
init_db(GRADES_DB, ["user", "sub", "score", "date"])
init_db(NOTIFY_DB, ["sender", "message", "date"])

def load_data(path):
    try: return pd.read_csv(path)
    except: return pd.DataFrame()

# ==========================================
# 3. إعدادات الواجهة والترحيب الزمني (بدون لاق)
# ==========================================
st.set_page_config(page_title="منصة سند التعليمية", layout="wide", page_icon="🎓")

hour = datetime.now().hour
if 5 <= hour < 12: time_greeting = "صباح الخير ☀️"
elif 12 <= hour < 18: time_greeting = "طاب نهارك 🌤️"
else: time_greeting = "مساء الخير 🌙"

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stButton>button { 
        width: 100%; border-radius: 8px; background: #1E88E5; color: white; 
        font-weight: bold; border: none; padding: 0.5rem; transition: 0.2s;
    }
    .stButton>button:active { transform: scale(0.98); }
    .modern-box { 
        padding: 15px; background-color: rgba(30, 136, 229, 0.05); 
        border-radius: 10px; border-right: 4px solid #1E88E5; margin-bottom: 15px;
    }
    .welcome-title { font-size: 1.8rem; font-weight: bold; text-align: center; color: #1E88E5; }
    .programmer-tag { font-size: 0.85rem; text-align: center; font-weight: bold; opacity: 0.7; }
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

if "user_data" not in st.session_state: st.session_state["user_data"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "oral_exam_history" not in st.session_state: st.session_state["oral_exam_history"] = []

# ==========================================
# نظام تسجيل الدخول التلقائي (حفظ الجلسة عند التحديث)
# ==========================================
if st.session_state["user_data"] is None and "session_token" in st.query_params:
    token = st.query_params["session_token"]
    if token == "Hosam":
        st.session_state["user_data"] = {"user": "Hosam", "role": "Owner", "grade": "الكل"}
    else:
        users = load_data(USERS_DB)
        match = users[users["user"] == token]
        if not match.empty:
            st.session_state["user_data"] = match.iloc[0].to_dict()

# ==========================================
# 4. شاشة الدخول والتسجيل
# ==========================================
if st.session_state["user_data"] is None:
    st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting}، مرحباً في سند</div><div class="programmer-tag">💻 برمجة الأستاذ حسام الأسدي</div></div>', unsafe_allow_html=True)
    
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب طالب"])
    
    with t_log:
        # استخدام st.form لتحفيز الموبايل على حفظ كلمة السر
        with st.form("login_form"):
            st.markdown("### 🔑 تسجيل الدخول")
            u = st.text_input("الاسم الكامل")
            p = st.text_input("كلمة المرور", type="password")
            submit = st.form_submit_button("دخول المنصة 🚀")
            
            if submit:
                if u == "Hosam" and p == "hosam031007":
                    st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                    st.query_params["session_token"] = u # حفظ الجلسة التلقائي
                    st.rerun()
                else:
                    users = load_data(USERS_DB)
                    if not users.empty:
                        match = users[(users["user"] == u) & (users["pass"] == hash_password(p))]
                        if not match.empty:
                            st.session_state["user_data"] = match.iloc[0].to_dict()
                            st.query_params["session_token"] = u # حفظ الجلسة التلقائي
                            st.rerun()
                        else: st.error("⚠️ عذراً، البيانات غير صحيحة")
                    else: st.warning("لا يوجد مستخدمين مسجلين بعد.")
    
    with t_sign:
        st.markdown("### 📋 بيانات الطالب الجديد")
        nu = st.text_input("الاسم الكامل (الرباعي)")
        ng = st.selectbox("الصف:", list(subs_map.keys()))
        fb = st.text_input("رابط حسابك على فيسبوك (للتوثيق 🌐)", placeholder="https://www.facebook.com/...")
        np = st.text_input("كلمة السر", type="password")
        np2 = st.text_input("تأكيد كلمة السر", type="password")
            
        if st.button("✅ تأكيد وإنشاء الحساب"):
            if not nu or not np or not np2 or not fb: st.warning("⚠️ يرجى تعبئة جميع الحقول.")
            elif np != np2: st.error("⚠️ كلمتا المرور غير متطابقتين.")
            elif "facebook.com" not in fb.lower() and "fb.com" not in fb.lower(): st.error("⚠️ يرجى إدخال رابط فيسبوك صحيح.")
            else:
                users = load_data(USERS_DB)
                if not users.empty and nu in users['user'].values: st.error("⚠️ الاسم موجود مسبقاً.")
                else:
                    new_user = pd.DataFrame([{"user": nu, "pass": hash_password(np), "role": "طالب", "grade": ng, "fb_link": fb}])
                    pd.concat([users, new_user], ignore_index=True).to_csv(USERS_DB, index=False)
                    st.success("🎉 تم إنشاء الحساب! سجل دخولك الآن.")

# ==========================================
# 5. شاشات المستخدمين (بعد تسجيل الدخول)
# ==========================================
else:
    user = st.session_state["user_data"]
    
    # --- القائمة الجانبية (Sidebar) ---
    with st.sidebar:
        profile_path = f"profiles/{user['user']}.png"
        if os.path.exists(profile_path):
            c1, c2, c3 = st.columns([1, 2, 1])
            c2.image(profile_path, use_container_width=True)
        else: st.markdown("<h1 style='text-align: center; color: #1E88E5;'>👤</h1>", unsafe_allow_html=True)
            
        st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{user['user']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray; font-weight: bold;'>{user['role']}</p>", unsafe_allow_html=True)
        if user['role'] == "طالب": st.markdown(f"<p style='text-align: center; color: #D32F2F;'>الصف: {user['grade']}</p>", unsafe_allow_html=True)
            
        st.divider()
        st.markdown("### 💎 حالة الحساب")
        if user['role'] == "Owner": st.success("حساب إدارة (VIP) 👑")
        elif user['role'] == "أستاذ": st.info("حساب كادر تدريسي 👨‍🏫")
        else:
            st.info("الخطة الحالية: مجانية 🆓")
            if st.button("🚀 الترقية لنسخة الـ PRO", use_container_width=True): st.toast("نظام الاشتراكات قريباً!")
                
        st.divider()
        st.markdown("### 🤝 دعوة صديق")
        st.text_input("انسخ الرابط وشاركه:", value="https://sanad.streamlit.app", disabled=True)
        if st.button("📋 نسخ رابط المنصة"): st.toast("تم النسخ! أرسله لأصدقائك.")
            
        st.divider()
        if st.button("🔴 تسجيل الخروج"):
            st.session_state["user_data"] = None
            if "session_token" in st.query_params: del st.query_params["session_token"]
            st.rerun()

    # ----------------------------------------
    # واجهة الإدارة (Owner Dashboard)
    # ----------------------------------------
    if user["role"] == "Owner":
        st.header(f"👑 لوحة تحكم الإدارة الشاملة - {time_greeting}")
        t_users, t_teachers, t_files, t_notify, t_settings = st.tabs(["👥 الطلاب", "👨‍🏫 الأساتذة", "📁 الملفات", "📩 رسائل الأساتذة", "⚙️ إعداداتي"])
        
        with t_users:
            u_df = load_data(USERS_DB)
            students = u_df[u_df['role'] == 'طالب']
            st.data_editor(students, num_rows="dynamic", use_container_width=True)

        with t_teachers:
            st.markdown("### ➕ إضافة أستاذ جديد")
            col1, col2 = st.columns(2)
            t_name = col1.text_input("اسم الأستاذ")
            t_pass = col2.text_input("كلمة مرور الأستاذ", type="password")
            if st.button("إنشاء حساب الأستاذ"):
                if t_name and t_pass:
                    users = load_data(USERS_DB)
                    if not users.empty and t_name in users['user'].values: st.error("الاسم موجود.")
                    else:
                        new_t = pd.DataFrame([{"user": t_name, "pass": hash_password(t_pass), "role": "أستاذ", "grade": "الكل", "fb_link": "معلم"}])
                        pd.concat([users, new_t], ignore_index=True).to_csv(USERS_DB, index=False)
                        st.success("تم تفعيل حساب الأستاذ بنجاح!")
                        st.rerun()
            st.markdown("---")
            st.markdown("### سجل الأساتذة")
            teachers_df = u_df[u_df['role'] == 'أستاذ']
            st.dataframe(teachers_df, use_container_width=True)

        with t_files:
            f_df = load_data(FILES_DB)
            file_to_del = st.selectbox("اختر الملف للحذف نهائياً:", [""] + list(f_df['name'].values))
            if st.button("🗑️ حذف الملف") and file_to_del:
                row = f_df[f_df['name'] == file_to_del].iloc[0]
                target_path = os.path.join("lessons" if row['type'] == "بحث" else "exams", file_to_del)
                if os.path.exists(target_path): os.remove(target_path)
                f_df[f_df['name'] != file_to_del].to_csv(FILES_DB, index=False)
                st.success("تم الحذف!")
                st.rerun()

        with t_notify:
            n_df = load_data(NOTIFY_DB)
            if not n_df.empty:
                st.dataframe(n_df, use_container_width=True)
                if st.button("مسح جميع التنويهات"): 
                    pd.DataFrame(columns=["sender", "message", "date"]).to_csv(NOTIFY_DB, index=False)
                    st.rerun()
            else: st.info("لا يوجد رسائل أو تنويهات جديدة من الأساتذة.")
                
        with t_settings:
            pic = st.file_uploader("ارفع صورتك الشخصية (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
            if pic and st.button("💾 حفظ الصورة"):
                Image.open(pic).save(f"profiles/{user['user']}.png")
                st.success("تم التحديث!")
                st.rerun()

    # ----------------------------------------
    # واجهة الطالب والأستاذ المشتركة 
    # ----------------------------------------
    elif user["role"] in ["طالب", "أستاذ"]:
        if user["role"] == "أستاذ":
            st.markdown(f'<div class="modern-box"><div class="welcome-title">👨‍🏫 أهلاً بك أستاذ {user["user"]}</div><div class="programmer-tag">اختر الصف والمادة لاستعراض المنصة أو رفع الدروس</div></div>', unsafe_allow_html=True)
            col_g, col_s = st.columns(2)
            view_grade = col_g.selectbox("تصفح كـ صف:", list(subs_map.keys()))
            sub = col_s.selectbox("المادة:", subs_map[view_grade])
            tabs = st.tabs(["📤 مركز الرفع", "📚 المكتبة", "🤖 المعلم الذكي", "📸 عدسة الذكاء", "📝 الامتحانات", "📅 المنقذ", "💬 مراسلة الإدارة"])
        else:
            st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting} يا بطل!</div><div class="programmer-tag">الصف: {user["grade"]}</div></div>', unsafe_allow_html=True)
            view_grade = user["grade"]
            sub = st.selectbox("اختر المادة التي ترغب بدراستها:", subs_map[view_grade])
            tabs = st.tabs(["📚 المكتبة", "🤖 المعلم الذكي", "📸 عدسة الذكاء", "📝 الامتحانات", "📅 المنقذ", "📊 مستواي"])

        # توزيع التبويبات حسب الصلاحية لتجنب تكرار الكود
        tab_index = 0

        # -- تاب الرفع (فقط للأستاذ) --
        if user["role"] == "أستاذ":
            with tabs[tab_index]:
                st.info("أرسل ملف PDF إلى بوت التلغرام، ثم اضغط جلب لرفعه لطلاب هذا الصف.")
                if st.button("🔄 جلب من التلغرام"):
                    updates = get_telegram_updates(BOT_TOKEN)
                    docs = [{"id": u["message"]["document"]["file_id"], "name": u["message"]["document"].get("file_name", "ملف.pdf"), "date": datetime.fromtimestamp(u["message"]["date"]).strftime("%Y-%m-%d %H:%M")} for u in updates if "message" in u and "document" in u["message"] and u["message"]["document"].get("mime_type") == "application/pdf"]
                    if docs: st.session_state["tg_docs"] = docs[-10:]; st.success("تم العثور على ملفات!")
                    else: st.warning("لا يوجد ملفات جديدة.")
                if st.session_state.get("tg_docs"):
                    selected_doc = {f"{d['name']} ({d['date']})": d for d in st.session_state["tg_docs"]}[st.selectbox("اختر الملف:", list({f"{d['name']} ({d['date']})": d for d in st.session_state["tg_docs"]}.keys()))]
                    type_f = st.radio("نوعه:", ["بحث", "نموذج امتحاني"])
                    if st.button("🚀 رفع الملف للمنصة"):
                        f_name = f"{type_f}_{sub}_{selected_doc['name'].replace(' ', '_')}"
                        if download_telegram_file(BOT_TOKEN, selected_doc['id'], os.path.join("lessons" if type_f=="بحث" else "exams", f_name)):
                            pd.concat([load_data(FILES_DB), pd.DataFrame([{"name": f_name, "grade": view_grade, "sub": sub, "type": type_f, "date": datetime.now().strftime("%Y-%m-%d")}])], ignore_index=True).to_csv(FILES_DB, index=False)
                            st.success("تم الرفع!")
            tab_index += 1

        # -- المكتبة --
        with tabs[tab_index]:
            search_q = st.text_input("🔍 ابحث عن درس...")
            f_db = load_data(FILES_DB)
            if not f_db.empty:
                my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)]
                if search_q: my_f = my_f[my_f['name'].str.contains(search_q, case=False)]
                if my_f.empty: st.info("لا توجد ملفات.")
                else:
                    for _, r in my_f.iterrows():
                        path = os.path.join("lessons" if r['type'] == "بحث" else "exams", r['name'])
                        if os.path.exists(path):
                            with open(path, "rb") as f: st.download_button(f"📥 {r['name'].split('_')[-1]}", f, file_name=r['name'], key=r['name'])
            else: st.info("المكتبة فارغة.")
        tab_index += 1

        # -- المعلم الذكي --
        with tabs[tab_index]:
            style = st.radio("طريقة الشرح:", ["علمي", "بالمشرمحي (ابن البلد)", "واقع سوري"], horizontal=True)
            for msg in st.session_state["chat_history"]: st.chat_message(msg["role"]).write(msg["content"])
            if q := st.chat_input("اكتب سؤالك..."):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                with st.spinner("يجهز الإجابة..."):
                    pr = f"أنت خبير سوري. أجب لمادة {sub} صف {view_grade}: {q}\n" + ("اشرحها عامية سورية بأمثلة من الشارع" if style=="بالمشرمحي (ابن البلد)" else "اربطها بواقع سوريا اليومي" if style=="واقع سوري" else "")
                    ans = get_ai_response(pr)
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)
        tab_index += 1

        # -- عدسة الذكاء --
        with tabs[tab_index]:
            v_mode = st.radio("الخدمة:", ["شرح مسألة", "تصحيح حل طالب"])
            if img := st.file_uploader("ارفع الصورة", type=["jpg", "png", "jpeg"]):
                if st.button("🚀 تحليل"):
                    with st.spinner("جاري التحليل..."):
                        res = get_ai_response(f"أنت معلم لمادة {sub} لصف {view_grade}. " + ("اشرح الدرس وطريقة الحل للمسألة في الصورة خطوة بخطوة ولا تعط الجواب فورا." if v_mode=="شرح مسألة" else "صحح الحل بالصورة وحدد الخطأ بدقة كأنك ترسم دائرة حمراء عليه، وأعط درجة من 100."), Image.open(img))
                        st.info(res)
        tab_index += 1

        # -- الامتحانات --
        with tabs[tab_index]:
            if st.radio("النوع:", ["📝 نموذج شامل", "🗣️ شفهي"]) == "📝 نموذج شامل":
                if st.button("🎯 توليد وزاري"): st.markdown(f'<div class="modern-box">{get_ai_response(f"ولد نموذج وزاري سوري لمادة {sub} صف {view_grade}.")}</div>', unsafe_allow_html=True)
            else:
                for m in st.session_state["oral_exam_history"]: st.chat_message(m["role"]).write(m["content"])
                if oral_q := st.chat_input("إجابتك..."):
                    st.session_state["oral_exam_history"].append({"role": "user", "content": oral_q})
                    st.chat_message("user").write(oral_q)
                    with st.spinner("يقيّم..."):
                        o_ans = get_ai_response(f"صحح إجابة الطالب: '{oral_q}' بمادة {sub}، واطرح سؤال شفهي جديد.")
                    st.session_state["oral_exam_history"].append({"role": "assistant", "content": o_ans})
                    st.chat_message("assistant").write(o_ans)
        tab_index += 1

        # -- المنقذ --
        with tabs[tab_index]:
            ca, cb = st.columns(2)
            if st.button("توليد الخطة"): st.markdown(f'<div class="modern-box">{get_ai_response(f"خطة دراسة {sub} لصف {view_grade} في {ca.number_input("أيام؟",1,value=7)} أيام بـ {cb.slider("ساعات؟",1,15,5)} ساعات يوميا.")}</div>', unsafe_allow_html=True)
        tab_index += 1

        # -- التاب الأخير (مستواي للطالب / مراسلة للإدارة للأستاذ) --
        with tabs[tab_index]:
            if user["role"] == "طالب":
                my_s = load_data(GRADES_DB)
                my_s = my_s[(my_s["user"] == user["user"]) & (my_s["sub"] == sub)]
                if not my_s.empty: st.line_chart(my_s.set_index("date")["score"]); st.metric("متوسط درجاتك", f"{my_s['score'].mean():.1f}%")
                else: st.info("لا درجات مسجلة.")
            else:
                st.markdown("### 💬 إرسال تنويه لمالك المنصة")
                msg = st.text_area("اكتب رسالتك، استفسارك، أو تقريرك هنا:")
                if st.button("إرسال للإدارة 📩") and msg:
                    n_db = load_data(NOTIFY_DB)
                    new_n = pd.DataFrame([{"sender": user["user"], "message": msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])
                    pd.concat([n_db, new_n], ignore_index=True).to_csv(NOTIFY_DB, index=False)
                    st.success("تم إرسال رسالتك للأستاذ حسام بنجاح!")
