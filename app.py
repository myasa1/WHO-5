import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# 1️⃣ إنشاء بيانات نفسية أكثر واقعية
# ----------------------------
@st.cache_data
def train_model():
    np.random.seed(42)
    n = 2000

    Q = np.random.randint(0, 6, (n, 5))
    sleep = np.random.randint(4, 11, n)
    activity = np.random.randint(1, 6, n)

    total = Q.sum(axis=1)

    # تصنيف ثلاثي أذكى
    risk = []
    for t in total:
        if t <= 10:
            risk.append(2)  # مرتفع
        elif t <= 17:
            risk.append(1)  # متوسط
        else:
            risk.append(0)  # منخفض

    X = np.column_stack((Q, sleep, activity))
    y = np.array(risk)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

# ----------------------------
# 2️⃣ واجهة المستخدم
# ----------------------------
st.title("🧠 تقييم ذكي للرفاهية النفسية (ML Version Advanced)")

q1 = st.slider("المزاج الجيد", 0, 5, 2)
q2 = st.slider("الهدوء والاسترخاء", 0, 5, 2)
q3 = st.slider("النشاط والحيوية", 0, 5, 2)
q4 = st.slider("النوم المنتعش", 0, 5, 2)
q5 = st.slider("الإحساس بالمعنى", 0, 5, 2)

sleep = st.slider("عدد ساعات النوم", 4, 10, 7)
activity = st.slider("مستوى النشاط البدني", 1, 5, 3)

# ----------------------------
# 3️⃣ التحليل الذكي
# ----------------------------
if st.button("تحليل ذكي"):

    user_input = np.array([[q1, q2, q3, q4, q5, sleep, activity]])
    prediction = model.predict(user_input)[0]
    probabilities = model.predict_proba(user_input)[0]

    st.subheader("📊 نتائج التحليل")

    if prediction == 2:
        st.error("🔴 مستوى خطورة مرتفع")
    elif prediction == 1:
        st.warning("🟡 مستوى متوسط")
    else:
        st.success("🟢 مستوى جيد")

    st.write(f"احتمالية منخفض: {probabilities[0]*100:.1f}%")
    st.write(f"احتمالية متوسط: {probabilities[1]*100:.1f}%")
    st.write(f"احتمالية مرتفع: {probabilities[2]*100:.1f}%")

    # تحليل أهمية العوامل
    features = ["مزاج", "هدوء", "طاقة", "نوم منتعش", "معنى", "ساعات النوم", "النشاط"]
    importance = model.feature_importances_
    top_factor = features[np.argmax(importance)]

    st.subheader("🔎 العامل الأكثر تأثيراً عليك:")
    st.write(f"العامل الأبرز في تقييمك حالياً هو: **{top_factor}**")

    # توصية ذكية مبنية على العامل
    st.subheader("📅 توصية مخصصة")

    if top_factor == "ساعات النوم":
        st.write("ركز على تنظيم النوم، تقليل الضوء الأزرق ليلاً، وثبات مواعيد النوم.")
    elif top_factor == "النشاط":
        st.write("زيادة النشاط البدني حتى 30 دقيقة يومياً قد تحسن حالتك.")
    else:
        st.write(f"يبدو أن جانب {top_factor} يحتاج دعم سلوكي مركز خلال الأسابيع القادمة.")

    st.info("هذا نموذج تعلم آلي تجريبي ولا يعد تشخيصاً طبياً.")
