# ==========================================
# IMPORTS
# ==========================================
import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import ssl
import certifi
from pdf2image import convert_from_bytes

# ==========================================
# SSL FIX
# ==========================================
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

# ==========================================
# OCR INIT
# ==========================================
reader = easyocr.Reader(['en'])

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Industrial CAD AI System",
    layout="wide"
)

st.title("🧠 Industrial CAD AI Validation System")

# ==========================================
# FILE UPLOADER
# ==========================================
uploaded_file = st.file_uploader(
    "Upload Image / PDF / Scanned Drawing",
    type=["png", "jpg", "jpeg", "pdf"]
)

# ==========================================
# SESSION STATE
# ==========================================
if "corrected_img" not in st.session_state:
    st.session_state.corrected_img = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ==========================================
# IMAGE PREPROCESS FUNCTION
# ==========================================
def preprocess_image(img):

    # --------------------------------------
    # GRAYSCALE
    # --------------------------------------
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------
    # DENOISE
    # --------------------------------------
    denoise = cv2.fastNlMeansDenoising(
        gray,
        None,
        30,
        7,
        21
    )

    # --------------------------------------
    # SHARPEN
    # --------------------------------------
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpen = cv2.filter2D(
        denoise,
        -1,
        kernel
    )

    # --------------------------------------
    # ADAPTIVE THRESHOLD
    # --------------------------------------
    thresh = cv2.adaptiveThreshold(
        sharpen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return gray, thresh

# ==========================================
# MAIN
# ==========================================
if uploaded_file:

    # --------------------------------------
    # RESET IF NEW FILE
    # --------------------------------------
    if (
        st.session_state.last_uploaded_file
        != uploaded_file.name
    ):

        st.session_state.corrected_img = None

        st.session_state.last_uploaded_file = (
            uploaded_file.name
        )

    # --------------------------------------
    # FILE TYPE
    # --------------------------------------
    file_type = uploaded_file.name.split(".")[-1].lower()

    # --------------------------------------
    # PDF SUPPORT
    # --------------------------------------
    if file_type == "pdf":

        pages = convert_from_bytes(
            uploaded_file.read()
        )

        image = pages[0]

    else:

        image = Image.open(uploaded_file)

    # --------------------------------------
    # IMAGE CONVERSION
    # --------------------------------------
    img_rgb = np.array(image)

    img = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------
    # INITIALIZE CORRECTED IMAGE
    # --------------------------------------
    if (
        st.session_state.corrected_img is None
    ):

        st.session_state.corrected_img = (
            img.copy()
        )

    # --------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------
    gray, thresh = preprocess_image(img)

    # --------------------------------------
    # OCR
    # --------------------------------------
    results = reader.readtext(thresh)

    detected_text = []

    # ======================================
    # TABS
    # ======================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 OCR",
        "⚠ Validation",
        "🛠 Auto Correction",
        "📊 Final Output"
    ])

    # ======================================
    # TAB 1 — OCR
    # ======================================
    with tab1:

        st.subheader("Original Input")

        st.image(
            image,
            use_container_width=True
        )

        st.subheader("Preprocessed Scan")

        st.image(
            thresh,
            use_container_width=True
        )

        st.subheader("OCR Detection")

        for r in results:

            bbox, text, conf = r

            detected_text.append(text)

            st.write(
                f"TEXT: {text} | "
                f"Confidence: {conf:.2f}"
            )

    # ======================================
    # GEOMETRY DETECTION
    # ======================================
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hole_count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if 200 < area < 5000:

            perimeter = cv2.arcLength(
                cnt,
                True
            )

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * area /
                (perimeter * perimeter)
            )

            if 0.7 < circularity < 1.2:

                hole_count += 1

    # ======================================
    # VALIDATION ENGINE
    # ======================================
    issues = []

    issue_id = 1

    joined_text = " ".join(
        detected_text
    ).upper()

    # ======================================
    # FONT SIZE CHECK
    # ======================================
    for r in results:

        bbox, text, conf = r

        height = abs(
            int(bbox[2][1]) -
            int(bbox[0][1])
        )

        if height < 20:

            issues.append({
                "id": issue_id,
                "type": "font",
                "text": text,
                "bbox": bbox,
                "message": "Font size too small",
                "suggestion": "Increase font size"
            })

            issue_id += 1

    # ======================================
    # TOLERANCE CHECK
    # ======================================
    tolerance_found = False

    for t in detected_text:

        t_clean = t.upper()

        if (
            "±" in t_clean
            or "+0.01" in t_clean
            or "+0.02" in t_clean
        ):

            tolerance_found = True

    if not tolerance_found:

        issues.append({
            "id": issue_id,
            "type": "tolerance",
            "message": "Missing tolerance",
            "suggestion": "Add tolerance ±0.01"
        })

        issue_id += 1

    # ======================================
    # HOLE SIZE CHECK
    # ======================================
    for r in results:

        bbox, text, conf = r

        t = text.upper().strip()

        if (
            t == "8"
            or "Ø8" in t
            or "08" in t
        ):

            issues.append({
                "id": issue_id,
                "type": "hole",
                "message": "Hole size below standard",
                "suggestion": "Use minimum Ø10"
            })

            issue_id += 1

    # ======================================
    # GEOMETRY CHECK
    # ======================================
    if hole_count == 0:

        issues.append({
            "id": issue_id,
            "type": "geometry",
            "message": "No holes detected",
            "suggestion": "Verify geometry"
        })

        issue_id += 1

    # ======================================
    # TAB 2 — VALIDATION
    # ======================================
    with tab2:

        st.subheader("Validation Issues")

        if len(issues) == 0:

            st.success(
                "✔ No Issues Found"
            )

        else:

            for issue in issues:

                st.error(
                    f"{issue['id']}. "
                    f"{issue['message']}"
                )

                st.write(
                    f"Suggested Fix: "
                    f"{issue['suggestion']}"
                )

    # ======================================
    # TAB 3 — AUTO CORRECTION
    # ======================================
    with tab3:

        st.subheader(
            "Interactive Auto Correction"
        )

        font_issues = [
            i for i in issues
            if i["type"] == "font"
        ]

        if len(font_issues) == 0:

            st.success(
                "✔ No Auto-Correctable Issues"
            )

        else:

            for issue in font_issues:

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.write(
                        f"Issue "
                        f"{issue['id']}"
                    )

                    st.write(
                        f"Error: "
                        f"{issue['message']}"
                    )

                    st.write(
                        f"Fix: "
                        f"{issue['suggestion']}"
                    )

                with col2:

                    if st.button(
                        f"Auto Correct "
                        f"{issue['id']}"
                    ):

                        bbox = issue["bbox"]

                        text = issue["text"]

                        top_left = tuple(
                            map(int, bbox[0])
                        )

                        bottom_right = tuple(
                            map(int, bbox[2])
                        )

                        # Remove old text
                        cv2.rectangle(
                            st.session_state.corrected_img,
                            top_left,
                            bottom_right,
                            (255, 255, 255),
                            -1
                        )

                        # Add corrected text
                        cv2.putText(
                            st.session_state.corrected_img,
                            text,
                            (
                                top_left[0],
                                top_left[1] + 40
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            3
                        )

                        st.success(
                            f"Issue "
                            f"{issue['id']} corrected"
                        )

            # ----------------------------------
            # LIVE PREVIEW
            # ----------------------------------
            st.subheader(
                "Live Corrected Drawing"
            )

            st.image(
                cv2.cvtColor(
                    st.session_state.corrected_img,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

    # ======================================
    # TAB 4 — FINAL OUTPUT
    # ======================================
    with tab4:

        st.subheader(
            "Final Corrected Drawing"
        )

        st.image(
            cv2.cvtColor(
                st.session_state.corrected_img,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        st.metric(
            "Total Issues Found",
            len(issues)
        )

        st.metric(
            "Detected Holes",
            hole_count
        )

        # --------------------------------------
        # RESET BUTTON
        # --------------------------------------
        if st.button("Reset Corrections"):

            st.session_state.corrected_img = (
                img.copy()
            )

            st.success(
                "Corrections reset successfully"
            )

# ==========================================
# NO FILE
# ==========================================
else:

    st.info(
        "Upload image, PDF or scanned drawing"
    )