import streamlit as st
import google.generativeai as genai
import PyPDF2
from gtts import gTTS
from PIL import Image
import io

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(
    page_title="المعلم الذكي الشامل",
    page_icon="🎓",
    layout="wide"
)

# --- 2. إعداد المفتاح ---
# المفتاح مدمج مباشرة بالكود كما طلبت
API_KEY = "AIzaSyCn33VD-Dc241aVPEkh7HuSQRw0K1fHGB4"
genai.configure(api_key=API_KEY)

# --- 3. الدوال المساعدة ---

# دالة قراءة النصوص من ملف PDF
def get_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        return f"خطأ في قراءة الـ PDF: {e}"

# دالة تحويل النص إلى صوت
def text_to_speech(text):
    try:
        # تنظيف النص من الرموز المزعجة للنطق
        clean_text = text[:250].replace("*", "").replace("#", "").replace("-", "")
        if clean_text:
            tts = gTTS(text=clean_text, lang='ar')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
    except Exception as e:
        return None
    return None

# --- 4. إعداد الموديل ---
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 5. واجهة المستخدم (الشريط الجانبي) ---
with st.sidebar:
    st.title("📂 أدوات الطالب الذكي")
    st.markdown("---")
    uploaded_pdf = st.file_uploader("1️⃣ ارفع كتابك (PDF)", type=['pdf'])
    uploaded_image = st.file_uploader("2️⃣ صور مسألة أو صفحة", type=['jpg', 'jpeg', 'png'])
    
    st.markdown("---")
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# --- 6. معالجة البيانات المرفوعة ---
context_text = ""
image_part = None

if uploaded_pdf:
    with st.spinner('جارٍ تحليل الكتاب...'):
        context_text = get_pdf_text(uploaded_pdf)
    st.sidebar.success("✅ الكتاب جاهز للتحليل")

if uploaded_image:
    image_part = Image.open(uploaded_image)
    st.sidebar.image(image_part, caption="الصورة التي تم رفعها")

# --- 7. واجهة المحادثة الرئيسية ---
st.title("🎓 المعلم الذكي")
st.write("أنا مساعدك الشخصي. يمكنني شرح الدروس من كتبك، حل المسائل من الصور، أو الإجابة على أي سؤال عام.")

# تهيئة سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# منطقة إدخال السؤال
if prompt := st.chat_input("بماذا يمكنني مساعدتك اليوم؟"):
    
    # إضافة سؤال المستخدم للسجل وعرضه
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # استجابة المعلم الذكي
    with st.chat_message("assistant"):
        with st.spinner('جاري التفكير...'):
            try:
                # الحالة 1: يوجد صورة مع سؤال
                if image_part:
                    response = model.generate_content([prompt, image_part])
                
                # الحالة 2: يوجد كتاب PDF مع سؤال
                elif context_text:
                    full_prompt = f"بناءً على المعلومات الموجودة في هذا الكتاب: \n{context_text[:10000]}\n\nأجب على سؤال الطالب بالتفصيل: {prompt}"
                    response = model.generate_content(full_prompt)
                
                # الحالة 3: سؤال عام بدون مرفقات
                else:
                    response = model.generate_content(prompt)
                
                res_text = response.text
                st.markdown(res_text)
                
                # إضافة خاصية الصوت
                audio = text_to_speech(res_text)
                if audio:
                    st.audio(audio, format="audio/mp3")
                
                # حفظ الإجابة في السجل
                st.session_state.messages.append({"role": "assistant", "content": res_text})
                
            except Exception as e:
                st.error(f"عذراً، حدث خطأ تقني: {e}")

# --- نهاية الكود ---
