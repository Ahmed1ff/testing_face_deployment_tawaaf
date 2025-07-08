import streamlit as st
import requests

st.set_page_config(page_title="Face Recognition", page_icon="📸")

st.title("📸 Face Recognition App")
st.markdown("Upload a photo and we'll try to match it with known faces from the database.")

# رفع الصورة
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Recognize Face"):
        with st.spinner("Processing..."):
            # إرسال الصورة إلى FastAPI
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://localhost:8000/upload_image", files=files)
                result = response.json()

                if response.status_code == 200 and result["found_faces"] == 1:
                    st.success(f"✅ Match found: **{result['matched_name']}**")
                    st.write(f"🔢 Distance: `{result['distance']:.4f}`")
                else:
                    st.warning("❌ No match found")

            except Exception as e:
                st.error(f"❌ Error connecting to API: {e}")
