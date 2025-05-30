import streamlit as st
import cv2
import numpy as np

import os
import io
import re

from pathlib import Path
from PIL import Image
from fuzzywuzzy import process
import requests


from word2number import w2n
from transformers import pipeline
import pandas as pd

st.title("Automated Grading System")

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

ip_address = st.text_input(
    label="Enter IP Webcam URL",
    placeholder="http://192.168.0.000:8080",
    value=""
)


def is_valid_url(url):
    pattern = re.compile(r'^http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$')
    return bool(pattern.match(url))


def get_shot_url(base_url):
    if not base_url.endswith("/shot.jpg"):
        base_url = base_url.rstrip("/") + "/shot.jpg"
    return base_url


def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None


desktop_path = Path.home() / "Desktop"
image_path = desktop_path / "captured_image.jpg"


st.subheader("Capture Image")
if st.button("Capture Image"):
    if not ip_address or not is_valid_url(ip_address):
        st.error("Please enter a valid IP Webcam URL (e.g., http://192.168.1.100:8080)")
    else:
        shot_url = get_shot_url(ip_address)
        img = fetch_image(shot_url)
        if img is not None:
            st.image(img, caption="Captured Image", use_container_width=True)


            if image_path.exists():
                os.remove(image_path)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), img_bgr)
            st.success(f"Image saved to Desktop as 'captured_image.jpg'!")


st.subheader("Upload Image for Processing")
uploaded_file = st.file_uploader(
    "Upload the Captured Image",
    type=["jpg", "jpeg"],
    accept_multiple_files=False,
    key="file_uploader"
)


def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_blur = cv2.medianBlur(gray_image, 3)
    _, result_image = cv2.threshold(median_blur, 180, 255, cv2.THRESH_BINARY)

    def noise_removal(image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    no_noise = noise_removal(result_image)

    temp_path = "temp_processed.jpg"
    cv2.imwrite(temp_path, no_noise)

    return img, temp_path


def detect_handwriting_contours(image_path):

    img = cv2.imread(image_path)
    if img is None:
        st.error(" Error: Image not loaded")
        return None


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)


    kernel = np.ones((5, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    min_area = 500
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h, area))
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if not boxes:
        st.warning("No valid boxes detected")
        return None

    largest_box = max(boxes, key=lambda b: b[4])
    lx, ly, lw, lh, _ = largest_box

    top_left_box = min(boxes, key=lambda b: b[0] + b[1])
    tx, ty, tw, th, _ = top_left_box

    RED1_HEIGHT_MULTIPLIER = 1.7
    RED2_HEIGHT_RATIO = 0.65
    RED2_X_OFFSET = 0.837
    RED_HEADER2_WIDTH_RATIO = 0.52
    RED_HEADER2_X_START = 1.0

    RED1_COLOR = (0, 0, 255)
    RED2_COLOR = (0, 0, 255)
    LARGEST_BOX_COLOR = (255, 0, 0)
    TOPLEFT_BOX_COLOR = (200, 0, 200)
    RED_HEADER2_COLOR = (0, 255, 255)


    red1_h = int(ty + RED1_HEIGHT_MULTIPLIER * th)
    red1_w = img.shape[1]
    cv2.rectangle(output, (0, 0), (red1_w, red1_h), RED1_COLOR, 2)

    red2_w = int(red1_w * RED_HEADER2_WIDTH_RATIO)
    red2_x = int(RED_HEADER2_X_START * (red1_w - red2_w))
    cv2.rectangle(output, (red2_x, 0), (red2_x + red2_w, red1_h), RED_HEADER2_COLOR, 2)

    red3_height = int(RED2_HEIGHT_RATIO * lh)
    red3_y = ly + lh - red3_height
    red3_x = lx + int(lw * RED2_X_OFFSET)

    right_edge = lx + lw
    bottom_edge = ly + lh

    cv2.rectangle(output, (red3_x, red3_y), (right_edge, bottom_edge), RED2_COLOR, 2)

    cv2.rectangle(output, (lx, ly), (lx + lw, ly + lh), LARGEST_BOX_COLOR, 3)
    cv2.rectangle(output, (tx, ty), (tx + tw, ty + th), TOPLEFT_BOX_COLOR, 3)

    OUTPUT_DIR = "output_coords"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cv2.imwrite(f"{OUTPUT_DIR}/red_header.jpg", img[0:red1_h, 0:red1_w])
    cv2.imwrite(f"{OUTPUT_DIR}/red_header2.jpg", img[0:red1_h, red2_x:red2_x + red2_w])
    cv2.imwrite(f"{OUTPUT_DIR}/red_lower.jpg", img[red3_y:bottom_edge, red3_x:right_edge])

    st.subheader("Roll Number and Marks Detected Image")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.success("Processing complete!")

    st.subheader("Cropped Regions")
    st.image(f"{OUTPUT_DIR}/red_header.jpg", caption="Roll Number", use_container_width=True, width=300)
    st.image(f"{OUTPUT_DIR}/red_header2.jpg", caption="Roll Number in words", use_container_width=True, width=300)
    st.image(f"{OUTPUT_DIR}/red_lower.jpg", caption="Total Marks", use_container_width=True, width=300)

def halve_image_height(img):
    width, height = img.size
    new_height = height // 2
    img = img.resize((width, new_height))
    return img

if uploaded_file is not None:
    original_img, processed_path = process_image(uploaded_file)
    if st.button("Run Handwriting Detection"):
        detect_handwriting_contours(processed_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)

@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="microsoft/trocr-large-handwritten", use_fast=False)

def image_to_bytes(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@st.cache_data
def run_inference(image_bytes):
    model = load_model()
    image = Image.open(io.BytesIO(image_bytes))
    return model(image)

if uploaded_file is not None and os.path.exists("output_coords/red_header.jpg"):
    try:
        header_img = Image.open("output_coords/red_header.jpg").convert("RGB")
        image_bytes1 = image_to_bytes(header_img)
        header_text = run_inference(image_bytes1)
        po_rf = header_text[0]['generated_text']

    except Exception as e:
        st.error(f"Header recognition failed: {str(e)}")

    try:
        sec_img = Image.open("output_coords/red_header2.jpg").convert("RGB")
        image_bytes2 = image_to_bytes(sec_img)
        header_text2 = run_inference(image_bytes2)
        po_rw = header_text2[0]['generated_text']

    except Exception as e:
        st.error(f"Secondary text recognition failed: {str(e)}")

    try:
        amount_img = Image.open("output_coords/red_lower.jpg").convert("RGB")
        image_bytes3 = image_to_bytes(amount_img)
        header_text3 = run_inference(image_bytes3)
        po_tm = header_text3[0]['generated_text']

    except Exception as e:
        st.error(f"Amount recognition failed: {str(e)}")

st.header("Text Recognition Result")

def extract_roll_number(text):
    replacements = {
        'O': '0', 'o': '0', 'Q': '0',
        'T': '1', 'l': '1', 'I': '1', 'F':'7'
    }
    idx = text.find("in words")
    if idx == -1 or idx < 1:
        return None

    start_idx = idx - 1
    digits = []
    i = start_idx - 1
    while i >= 0 and len(digits) < 3:
        ch = text[i]
        if ch == '.':
            i -= 1
            continue
        ch = replacements.get(ch, ch)
        if ch.isdigit():
            digits.insert(0, ch)
        i -= 1

    return ''.join(digits)

st.subheader("Roll Number Extraction")

generated_text = po_rf
roll_number = extract_roll_number(generated_text)

st.write("Final roll number from figure- ", roll_number)

def clean_textp(text):

    text = text.replace('.', ' ')
    text = text.replace('#', ' ')
    text = text.replace('@', ' ')
    text = text.replace('$', ' ')
    text = text.replace('%', ' ')
    text = text.replace('^', ' ')
    text = text.replace('&', ' ')
    text = text.replace('*', ' ')
    text = text.replace('-', ' ')

    text = re.sub(r'^.*?\)', '', text)
    return text.strip()

number_parts = [
    # Units
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    # Teens
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
    # Tens
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    # Hundreds & extras
    "hundred", "only", "and"
]

def correct_number_partp(part):
    if part in number_parts:
        return part
    best_match, score = process.extractOne(part, number_parts)
    return best_match if score > 70 else part

def correct_composite_numberp(text):
    parts = text.split()
    corrected_parts = [correct_number_partp(part) for part in parts]
    return " ".join(corrected_parts)

generated_text3 = po_rw
cleaned_text = clean_textp(generated_text3)

final_corrected_text = correct_composite_numberp(cleaned_text)

try:
    number = w2n.word_to_num(final_corrected_text)
    st.write(f"**Final Roll number from words-** {number}")
except ValueError:
    st.error("Could not convert the cleaned text to a number.")
    number= -1

st.subheader("Marks Extraction")
st.write("Total Marks-", po_tm)

st.subheader("Raw Data")
st.write("Raw Text roll number from figure- ", generated_text)
st.write("Raw Roll number in words-", po_rw)
#st.write("Raw Roll number in words-", cleaned_text)
#st.write("Raw Roll number in words-", final_corrected_text)

def clean_and_convert_to_digit(text):
    replacements = {
        'O': '0', 'o': '0', 'Q': '0',
        'l': '1', 'I': '1', 'T': '1'
    }

    cleaned = ''
    for ch in text:
        if ch.isdigit():
            cleaned += ch
        elif ch in replacements:
            cleaned += replacements[ch]

    return int(cleaned) if cleaned else None
marks = po_tm
txt1= str(number)
txt2= str(roll_number)
option = st.radio("Select an option for Roll number:", [txt1, txt2])

roll_number1 = None

if option == txt1:
    roll_number1 = number
elif option == txt2:
    roll_number1 = roll_number


if st.button("Submit"):
    st.write(roll_number1)
    if roll_number1 and marks is not None:
        num1 = str(roll_number1)
        num2 = marks
        filename = 'marks.xlsx'

        if os.path.exists(filename):
            df = pd.read_excel(filename)
            df['Roll number'] = df['Roll number'].astype(str)
        else:
            df = pd.DataFrame(columns=['Roll number', 'Marks'])

        new_row = {'Roll number': num1, 'Marks': num2}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_excel(filename, index=False)
        st.write("Marks updated successfully!")

        clean_df = pd.read_excel(filename)
        clean_df['Roll number'] = clean_df['Roll number'].astype(str)
        clean_df = clean_df[['Roll number', 'Marks']]
        clean_df = clean_df.dropna()
        clean_df.to_excel(filename, index=False)


        st.download_button(label="Download Excel file", data=open(filename, 'rb'), file_name=filename)

    else:
        st.error("Please provide both roll number and marks!")


filename = 'marks.xlsx'
st.subheader("View and Edit Last 10 Rows")
if os.path.exists(filename):
    view_df = pd.read_excel(filename)
    view_df['Roll number'] = view_df['Roll number'].astype(str)
    if not view_df.empty:

        displayed_df = view_df.tail(10)

        edited_df = st.data_editor(
            displayed_df,
            column_config={
                "Roll number": st.column_config.TextColumn(
                    "Roll number",
                    help="Edit roll number as text"
                ),
                "Marks": st.column_config.NumberColumn(
                    "Marks",
                    min_value=0,
                    max_value=100,
                    step=1,
                    help="Edit marks (0-100)"
                )
            },
            num_rows="fixed",
            disabled=False,
            key="data_editor_unique"
        )


        if st.button("Save Changes"):

            full_df = view_df.copy()
            edited_indices = edited_df.index
            full_df.loc[edited_indices, ['Roll number', 'Marks']] = edited_df[['Roll number', 'Marks']]

            full_df['Roll number'] = full_df['Roll number'].astype(str)
            full_df.to_excel(filename, index=False)
            st.success("Changes saved to Excel file!")
    else:
        st.info("No data in the Excel file yet.")
else:
    st.info("Excel file does not exist yet.")

if os.path.exists(filename):
    st.download_button(
        label="Download Excel File",
        data=open(filename, 'rb'),
        file_name=filename,
        key="download_always"
    )
else:
    st.info("No Excel file available to download.")

if st.button("Delete File"):
    if os.path.exists(filename):
        os.remove(filename)
        st.write("File deleted. Fresh start initiated!")
    else:
        st.error("File not found. Nothing to delete.")


if uploaded_file is not None and os.path.exists("temp_processed.jpg"):
    os.remove("temp_processed.jpg")