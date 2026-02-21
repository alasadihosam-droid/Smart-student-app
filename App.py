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
# 1. إعدادات الأمان، الذكاء الاصطناعي والتلغرام
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
        
        if not safe_models:
            return "⚠️ عذراً، جميع الموديلات المتاحة في حسابك غير مجانية أو محجوبة. تأكد من إعدادات حسابك."

        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                if image:
                    response = model.generate_content([prompt, image])
                else:
                    response = model.generate_content(prompt)
                return response.text 
            except Exception:
                continue 
                
        return "⚠️ تم رفض الاتصال من جوجل (نفاذ الرصيد أو حظر جغرافي). جرب تشغيل VPN."
    except Exception as e:
        return f"⚠️ خطأ عام في الاتصال: {str(e)}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except:
        return None

# ==========================================
# دوال التواصل مع بوت التلغرام
# ==========================================
def get_telegram_updates(token):
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        r = requests.get(url).json()
        if r.get("ok"):
            return r["result"]
    except:
        pass
    return []

def download_telegram_file(token, file_id, dest_path):
    file_info_url = f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}"
    try:
        r = requests.get(file_info_url).json()
        if r.get("ok"):
            file_path = r["result"]["file_path"]
            download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
            file_data = requests.get(download_url).content
            with open(dest_path, "wb") as f:
                f.write(file_data)
            return True
    except:
        pass
    return False

# ==========================================
# 2. تهيئة قواعد البيانات والمجلدات
# ==========================================
for folder in ['lessons', 'exams', 'db']:
    if not os.path.exists(folder):
        os.makedirs(folder)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
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

# ==========================================
# 3. إعدادات الواجهة والتصميم المحسن للموبايل
# ==========================================
st.set_page_config(page_title="منصة سند التعليمية", layout="wide", page_icon="🎓")

# تحسين الألوان وتخفيف الـ CSS لمنع التعليق
hour = datetime.now().hour
if 5 <= hour < 18:
    bg, txt, card_bg, border_c = "#F0F2F6", "#1E1E1E", "#FFFFFF", "#D32F2F"
else:
    bg, txt, card_bg, border_c = "#0E1117", "#FAFAFA", "#1E1E1E", "#FF5252"

st.markdown(f"""
    <style>
    /* تسريع الرندر وتخفيف الحمل على الموبايل */
    .stApp {{ 
        background-color: {bg}; 
        color: {txt} !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* فرض لون الخط على كل النصوص */
    p, h1, h2, h3, h4, h5, h6, span, div {{
        color: {txt} !important;
    }}
    
    .stButton>button {{ 
        width: 100%; 
        border-radius: 8px; 
        background-color: {border_c}; 
        color: white !important; 
        font-weight: bold; 
        border: none;
        transition: 0.2s;
    }}
    .stButton>button:active {{
        transform: scale(0.98);
    }}
    
    /* صندوق الترحيب الجديد */
    .greeting-box {{ 
        padding: 15px; 
        background-color: {card_bg}; 
        border-radius: 10px; 
        border-bottom: 3px solid {border_c}; 
        text-align: center; 
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    .greeting-title {{
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 5px;
        color: {txt} !important;
    }}
    
    .greeting-subtitle {{
        font-size: 1rem;
        color: {txt} !important;
        opacity: 0.8;
    }}
    
    .programmer-tag {{
        font-size: 0.85rem;
        color: {border_c} !important;
        font-weight: bold;
        margin-top: 10px;
    }}

    .admin-card, .exam-box {{
        padding: 15px; 
        background-color: {card_bg}; 
        border-left: 4px solid {border_c}; 
        border-radius: 5px; 
        margin-bottom: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

if "user_data" not in st.session_state:
    st.session_state["user_data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "oral_exam_history" not in st.session_state:
    st.session_state["oral_exam_history"] = []

# ==========================================
# 4. شاشة الدخول والتسجيل
# ==========================================
if st.session_state["user_data"] is None:
    st.markdown(f"""
        <div class="greeting-box">
            <div class="greeting-title">مرحباً في منصة سند التعليمية</div>
            <div class="programmer-tag">💻 برمجة الأستاذ حسام الأسدي</div>
        </div>
    """, unsafe_allow_html=True)
    
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب"])
    
    with t_log:
        login_col1, login_col2 = st.columns([1, 1])
        with login_col1:
            u = st.text_input("اسم المستخدم", key="login_u")
        with login_col2:
            p = st.text_input("كلمة المرور", type="password", key="login_p")
            
        if st.button("دخول المنصة"):
            if u == "Hosam" and p == "hosam031007":
                st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB)
                if not users.empty:
                    hashed_p = hash_password(p)
                    match = users[(users["user"] == u) & (users["pass"] == hashed_p)]
                    if not match.empty:
                        st.session_state["user_data"] = match.iloc[0].to_dict()
                        st.rerun()
                    else:
                        st.error("عذراً، البيانات غير صحيحة")
                else:
                    st.warning("لا يوجد مستخدمين مسجلين بعد.")
    
    with t_sign:
        sign_col1, sign_col2 = st.columns([1, 1])
        with sign_col1:
            nu = st.text_input("الاسم الكامل")
            nr = st.selectbox("أنا:", ["طالب", "أستاذ"])
        with sign_col2:
            np = st.text_input("كلمة السر", type="password")
            ng = st.selectbox("الصف:", list(subs_map.keys())) if nr == "طالب" else "الكل"
            
        if st.button("تأكيد إنشاء الحساب"):
            if nu and np:
                users = load_data(USERS_DB)
                if not users.empty and nu in users['user'].values:
                    st.error("الاسم موجود مسبقاً، يرجى اختيار اسم آخر.")
                else:
                    new_user = pd.DataFrame([{
                        "user": nu, "pass": hash_password(np), "role": nr, "grade": ng
                    }])
                    pd.concat([users, new_user], ignore_index=True).to_csv(USERS_DB, index=False)
                    st.success("تم إنشاء الحساب بنجاح! يمكنك تسجيل الدخول الآن.")
            else:
                st.warning("يرجى تعبئة جميع الحقول.")

# ==========================================
# 5. شاشات المستخدمين حسب الصلاحية
# ==========================================
else:
    user = st.session_state["user_data"]
    
    st.sidebar.markdown(f"### 👋 أهلاً {user['user']}")
    st.sidebar.markdown(f"**الصلاحية:** {user['role']}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🔴 تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.session_state["chat_history"] = []
        st.session_state["oral_exam_history"] = []
        st.rerun()

    # ----------------------------------------
    # واجهة الإدارة (Owner Dashboard)
    # ----------------------------------------
    if user["role"] == "Owner":
        st.header("👑 لوحة تحكم الإدارة الشاملة")
        t_users, t_files, t_all_grades = st.tabs(["👥 إدارة المستخدمين", "📁 إدارة الملفات", "📊 السجلات والدرجات"])
        
        with t_users:
            st.markdown('<div class="admin-card">هنا يمكنك عرض جميع الحسابات، وتعديل بياناتها، أو حذف أي مستخدم نهائياً.</div>', unsafe_allow_html=True)
            u_df = load_data(USERS_DB)
            del_col, edit_col = st.columns([1, 2])
            with del_col:
                user_to_del = st.selectbox("اختر اسم المستخدم للحذف:", [""] + list(u_df['user'].values))
                if st.button("🗑️ تأكيد الحذف") and user_to_del:
                    u_df = u_df[u_df['user'] != user_to_del]
                    u_df.to_csv(USERS_DB, index=False)
                    st.success(f"تم حذف {user_to_del}")
                    st.rerun()
            with edit_col:
                edited_u = st.data_editor(u_df, num_rows="dynamic", use_container_width=True)
                if st.button("💾 حفظ تعديلات المستخدمين"):
                    edited_u.to_csv(USERS_DB, index=False)
                    st.success("تم الحفظ!")

        with t_files:
            st.markdown('<div class="admin-card">لحذف ملف بالكامل (من قاعدة البيانات ومن السيرفر).</div>', unsafe_allow_html=True)
            f_df = load_data(FILES_DB)
            f_del_col, f_edit_col = st.columns([1, 2])
            with f_del_col:
                file_to_del = st.selectbox("اختر الملف للحذف:", [""] + list(f_df['name'].values))
                if st.button("🗑️ حذف الملف نهائياً") and file_to_del:
                    file_row = f_df[f_df['name'] == file_to_del].iloc[0]
                    folder = "lessons" if file_row['type'] == "بحث" else "exams"
                    target_path = os.path.join(folder, file_to_del)
                    if os.path.exists(target_path): os.remove(target_path)
                    f_df = f_df[f_df['name'] != file_to_del]
                    f_df.to_csv(FILES_DB, index=False)
                    st.success("تم تدمير الملف!")
                    st.rerun()
            with f_edit_col:
                edited_f = st.data_editor(f_df, num_rows="dynamic", use_container_width=True)
                if st.button("💾 حفظ تعديلات الملفات"):
                    edited_f.to_csv(FILES_DB, index=False)
                    st.success("تم الحفظ!")
                    
        with t_all_grades:
            g_df = load_data(GRADES_DB)
            edited_g = st.data_editor(g_df, num_rows="dynamic", use_container_width=True)
            if st.button("💾 حفظ تعديلات الدرجات"):
                edited_g.to_csv(GRADES_DB, index=False)
                st.success("تم الحفظ!")

    # ----------------------------------------
    # واجهة الأستاذ (الرفع عبر التلغرام)
    # ----------------------------------------
    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع الدروس (عبر التلغرام)")
        st.info("أرسل ملف PDF إلى بوت التلغرام، ثم اضغط 'جلب الملفات' لرفعه للمنصة.")
        
        if not BOT_TOKEN:
            st.warning("⚠️ ميزة التلغرام غير مفعلة. يرجى وضع TELEGRAM_BOT_TOKEN في الـ Secrets.")
        else:
            if st.button("🔄 جلب أحدث الملفات المرسلة للبوت"):
                with st.spinner("جاري الاتصال..."):
                    updates = get_telegram_updates(BOT_TOKEN)
                    docs = []
                    for u in updates:
                        if "message" in u and "document" in u["message"]:
                            doc = u["message"]["document"]
                            if doc.get("mime_type") == "application/pdf":
                                docs.append({
                                    "id": doc["file_id"],
                                    "name": doc.get("file_name", "ملف_بدون_اسم.pdf"),
                                    "date": datetime.fromtimestamp(u["message"]["date"]).strftime("%Y-%m-%d %H:%M")
                                })
                    if docs:
                        st.session_state["tg_docs"] = docs[-10:]
                        st.success("تم العثور على ملفات!")
                    else:
                        st.warning("لا يوجد ملفات PDF جديدة.")

            if st.session_state.get("tg_docs"):
                st.markdown("---")
                doc_dict = {f"{d['name']} ({d['date']})": d for d in st.session_state["tg_docs"]}
                selected_doc_name = st.selectbox("اختر الملف لرفعه:", list(doc_dict.keys()))
                selected_doc = doc_dict[selected_doc_name]
                
                c1, c2 = st.columns(2)
                tg = c1.selectbox("الصاف:", list(subs_map.keys()))
                ts = c2.selectbox("المادة:", subs_map[tg])
                type_f = st.radio("نوع الملف:", ["بحث", "نموذج امتحاني"])
                
                if st.button("🚀 سحب الملف للمنصة"):
                    f_name = f"{type_f}_{ts}_{selected_doc['name'].replace(' ', '_')}"
                    folder = "lessons" if type_f == "بحث" else "exams"
                    dest_path = os.path.join(folder, f_name)
                    
                    with st.spinner("جاري السحب..."):
                        if download_telegram_file(BOT_TOKEN, selected_doc['id'], dest_path):
                            f_db = load_data(FILES_DB)
                            new_file = pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_f, "date": datetime.now().strftime("%Y-%m-%d")}])
                            pd.concat([f_db, new_file], ignore_index=True).to_csv(FILES_DB, index=False)
                            st.success("تم الرفع بنجاح!")
                        else:
                            st.error("فشل السحب.")

    # ----------------------------------------
    # واجهة الطالب
    # ----------------------------------------
    elif user["role"] == "طالب":
        st.markdown(f"""
            <div class="greeting-box">
                <div class="greeting-title">مرحباً في منصة سند التعليمية</div>
                <div class="greeting-subtitle">أهلاً بك يا بطل | الصف: {user['grade']}</div>
                <div class="programmer-tag">💻 برمجة الأستاذ حسام الأسدي</div>
            </div>
        """, unsafe_allow_html=True)
        
        sub = st.selectbox("اختر المادة التي ترغب بدراستها:", subs_map[user['grade']])
        
        t_study, t_ai, t_vision, t_exams, t_plan, t_progress = st.tabs([
            "📚 المكتبة", 
            "🤖 المعلم الذكي", 
            "📸 عدسة الذكاء", 
            "📝 الامتحانات", 
            "📅 المنقذ", 
            "📊 مستواي"
        ])
        
        # 1. المكتبة
        with t_study:
            search_q = st.text_input("🔍 ابحث عن اسم درس...")
            f_db = load_data(FILES_DB)
            if not f_db.empty:
                my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
                if search_q: my_f = my_f[my_f['name'].str.contains(search_q, case=False)]
                if my_f.empty: st.info("لا توجد ملفات مرفوعة.")
                else:
                    for _, r in my_f.iterrows():
                        folder, path = ("lessons" if r['type'] == "بحث" else "exams"), ""
                        path = os.path.join(folder, r['name'])
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                st.download_button(label=f"📥 {r['name'].split('_')[-1]}", data=f, file_name=r['name'], key=r['name'])
            else: st.info("المكتبة فارغة.")

        # 2. المعلم الذكي (ابن البلد)
        with t_ai:
            st.subheader("💬 اسأل المعلم الذكي")
            style = st.radio("كيف ترغب أن يشرح لك المعلم؟", ["شرح علمي عادي", "شرح بالمشرمحي (ابن البلد 🇸🇾)", "ربط بالواقع السوري 🛠️"], horizontal=True)
            
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]): st.write(msg["content"])
            
            q = st.chat_input("اكتب سؤالك هنا...")
            if q:
                st.session_state["chat_history"].append({"role": "user", "content": q})
                with st.chat_message("user"): st.write(q)
                
                with st.spinner("المعلم يجهز الإجابة..."):
                    ai_prompt = f"أنت معلم خبير في سوريا. أجب عن هذا السؤال لمادة {sub} لصف {user['grade']}: {q}\n"
                    if style == "شرح بالمشرمحي (ابن البلد 🇸🇾)":
                        ai_prompt += "المطلوب: اشرح الفكرة باللهجة السورية العامية واستخدم أمثلة من الشارع لتسهيل الفهم."
                    elif style == "ربط بالواقع السوري 🛠️":
                        ai_prompt += "المطلوب: اشرح المفهوم بربطه بسيناريوهات من الواقع السوري اليومي لتكون الفكرة منطقية."
                        
                    ans = get_ai_response(ai_prompt)
                
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                with st.chat_message("assistant"):
                    st.write(ans)
                    audio = speak_text(ans)
                    if audio: st.audio(audio, format="audio/mp3")

        # 3. عدسة الذكاء الاصطناعي
        with t_vision:
            st.subheader("📸 عدسة الذكاء الاصطناعي")
            vision_mode = st.radio("اختر الخدمة:", ["البحث العكسي (كيف أحل هذه المسألة؟)", "المصحح الآلي المقارن (أين خطأي؟)"])
            img = st.file_uploader("ارفع صورة المسألة أو الحل", type=["jpg", "png", "jpeg"])
            
            if img and st.button("🚀 بدء التحليل البصري"):
                with st.spinner("جاري مسح الصورة وتحليلها..."):
                    img_opened = Image.open(img)
                    if vision_mode == "البحث العكسي (كيف أحل هذه المسألة؟)":
                        v_prompt = f"أنت معلم ذكي لمادة {sub} لصف {user['grade']}. اشرح 'الدرس أو القانون' للمسألة وعلمه 'طريقة الحل' خطوة بخطوة."
                        res = get_ai_response(v_prompt, img_opened)
                        st.info(res)
                    elif vision_mode == "المصحح الآلي المقارن (أين خطأي؟)":
                        v_prompt = f"أنت مصحح لمادة {sub}. حلل ورقة الطالب وقارنها بالحل النموذجي. حدد الخطأ بدقة واشرح كيف يصححه، وأعطه درجة من 100."
                        res = get_ai_response(v_prompt, img_opened)
                        st.info(res)
                        try:
                            match = re.search(r'\d+', res)
                            if match:
                                score = min(int(match.group()), 100)
                                pd.concat([load_data(GRADES_DB), pd.DataFrame([{"user": user['user'], "sub": sub, "score": score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])], ignore_index=True).to_csv(GRADES_DB, index=False)
                                st.toast(f"تم تسجيل النتيجة: {score}/100")
                        except: pass

        # 4. محاكي الامتحانات
        with t_exams:
            exam_mode = st.radio("اختر نوع الامتحان:", ["📝 نموذج وزاري شامل", "🗣️ تسميع شفهي"])
            
            if exam_mode == "📝 نموذج وزاري شامل":
                if st.button("🎯 توليد نموذج وزاري الآن"):
                    with st.spinner("جاري صياغة الأسئلة..."):
                        e_prompt = f"قم بتوليد نموذج امتحاني وزاري شامل لمادة {sub} لصف {user['grade']}. يحاكي النمط الوزاري السوري الحقيقي."
                        exam_paper = get_ai_response(e_prompt)
                    st.markdown(f'<div class="exam-box">{exam_paper}</div>', unsafe_allow_html=True)
                    
            elif exam_mode == "🗣️ تسميع شفهي":
                for m in st.session_state["oral_exam_history"]:
                    with st.chat_message(m["role"]): st.write(m["content"])
                oral_q = st.chat_input("أدخل إجابتك هنا...")
                if oral_q:
                    st.session_state["oral_exam_history"].append({"role": "user", "content": oral_q})
                    with st.chat_message("user"): st.write(oral_q)
                    with st.spinner("يقيّم إجابتك..."):
                        o_prompt = f"أنت ممتحن شفهي لمادة {sub}. صحح إجابة الطالب: '{oral_q}' علمياً، ثم اطرح سؤالاً جديداً."
                        o_ans = get_ai_response(o_prompt)
                    st.session_state["oral_exam_history"].append({"role": "assistant", "content": o_ans})
                    with st.chat_message("assistant"):
                        st.write(o_ans)
                        audio = speak_text(o_ans)
                        if audio: st.audio(audio, format="audio/mp3")

        # 5. المنقذ
        with t_plan:
            c_a, c_b = st.columns(2)
            d = c_a.number_input("أيام للامتحان؟", min_value=1, value=7)
            h = c_b.slider("ساعات الدراسة يومياً؟", 1, 15, 5)
            if st.button("توليد خطة الإنقاذ"):
                with st.spinner("جاري التصميم..."):
                    plan = get_ai_response(f"جدول دراسي في مادة {sub} لصف {user['grade']} للانتهاء في {d} أيام بـ {h} ساعات.")
                st.markdown(f'<div class="exam-box">{plan}</div>', unsafe_allow_html=True)

        # 6. مستواي
        with t_progress:
            g_db = load_data(GRADES_DB)
            my_scores = g_db[(g_db["user"] == user["user"]) & (g_db["sub"] == sub)]
            if not my_scores.empty:
                st.line_chart(my_scores.set_index("date")["score"])
                st.metric("متوسط درجاتك", f"{my_scores['score'].mean():.1f}%")
            else: st.info("لا درجات مسجلة.")
