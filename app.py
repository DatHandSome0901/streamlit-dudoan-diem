import streamlit as st
import pandas as pd
import joblib

# ================== CẤU HÌNH GIAO DIỆN ==================
st.set_page_config(
    page_title="Demo dự đoán điểm cuối kỳ",
    page_icon="📊",
    layout="centered"
)

st.title("📊 DỰ ĐOÁN ĐIỂM CUỐI KỲ HỌC SINH")


# ================== GIẢI THÍCH THANG ĐIỂM & THUỘC TÍNH ==================
with st.expander("📘 Giải thích thang điểm và các thuộc tính"):
    st.markdown("""
### 🔹 Thang điểm
- Điểm số được chấm theo **thang điểm 0 – 20**

### 🔹 Các thuộc tính đầu vào
- **studytime**: Thời gian học mỗi tuần  
  - 1: rất ít  
  - 2: trung bình  
  - 3: nhiều  
  - 4: rất nhiều  

- **failures**: Số lần trượt môn trước đó  

- **absences**: Số buổi nghỉ học  

- **G1**: Điểm kiểm tra kỳ 1 (thang điểm 20)  
- **G2**: Điểm kiểm tra kỳ 2 (thang điểm 20)  

### 🔹 Biến mục tiêu
- **G3**: Điểm cuối kỳ – giá trị cần dự đoán
""")

# ================== LOAD MÔ HÌNH ==================
@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model.pkl")

model = load_model()

# ================== NHẬP DỮ LIỆU ==================
st.subheader("📝 Nhập thông tin học sinh")

col1, col2 = st.columns(2)

with col1:
    studytime = st.text_input("Thời gian học mỗi tuần (1-4)", "2")
    failures = st.text_input("Số lần trượt môn", "0")
    G1 = st.text_input("Điểm kiểm tra kỳ 1", "10")

with col2:
    absences = st.text_input("Số buổi nghỉ học", "5")
    G2 = st.text_input("Điểm kiểm tra kỳ 2", "10")

# ================== DỰ ĐOÁN ==================
if st.button("🎯 DỰ ĐOÁN ĐIỂM CUỐI KỲ"):
    try:
        studytime = float(studytime)
        failures = float(failures)
        absences = float(absences)
        G1 = float(G1)
        G2 = float(G2)

        input_data = pd.DataFrame({
            "studytime": [studytime],
            "failures": [failures],
            "absences": [absences],
            "G1": [G1],
            "G2": [G2]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"📌 **Điểm cuối kỳ dự đoán: {prediction:.2f}"
        )

    except ValueError:
        st.error("❌ Vui lòng nhập **giá trị số hợp lệ** cho tất cả các ô!")

# ================== GHI CHÚ ==================
st.markdown("---")
st.caption(
    "🔎 Ghi chú: Mô hình được huấn luyện trước bằng phương pháp "
    "Hồi quy tuyến tính đa biến trên tập dữ liệu Student Performance Dataset."

)



