# === Install Required Libraries ===
!apt install tesseract-ocr -y
!apt install poppler-utils -y
!pip install pytesseract pdf2image transformers openai PyPDF2 python-docx docx2txt

# === Imports ===
from google.colab import files
from pathlib import Path
import PyPDF2
import docx2txt
import pytesseract
from pdf2image import convert_from_path
import os
import re
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import openai

# === Step 1: Upload Resume ===
print("📄 Please upload your resume (.pdf or .docx):")
uploaded = files.upload()

# === Step 2: Extract Text ===
def extract_text_from_scanned_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text

file_name = list(uploaded.keys())[0]
file_path = Path(file_name)
text = ""

try:
    if file_path.suffix == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text.strip():
            print("⚠️ Fallback to OCR (scanned PDF detected)...")
            text = extract_text_from_scanned_pdf(file_path)
    elif file_path.suffix == ".docx":
        text = docx2txt.process(file_path)
    else:
        raise ValueError("❌ Only .pdf or .docx files are supported.")
except Exception as e:
    print(f"Error reading resume file: {e}")

print("\n--- Extracted Text Preview ---\n")
print(text[:1000])

# === Step 3: Extract Skills ===
def extract_skills(text):
    skills_list = [
        'python', 'java', 'c++', 'linux', 'tcp/ip', 'sql', 'mongodb',
        'docker', 'kubernetes', 'rest api', 'tensorflow', 'pytorch',
        'data structures', 'algorithms', 'networking', 'git', 'aws'
    ]
    found = []
    for skill in skills_list:
        if skill.lower() in text.lower():
            found.append(skill)
    return list(set(found))

skills = extract_skills(text)
print("\n✅ Extracted Skills:", skills)

# === Step 4: Load BERT Model for QA ===
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
bert_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def get_answer_with_bert(question, context):
    try:
        inputs = bert_tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = bert_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = bert_tokenizer.convert_tokens_to_string(
            bert_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        return answer
    except Exception as e:
        print(f"Error with BERT model: {e}")
        return "Error generating answer with BERT."

# === Step 5: Configure OpenAI API Key ===
openai.api_key = "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"  # 🔐 Replace with your actual OpenAI API key

def generate_gpt_question(skill):
    try:
        prompt = f"Generate a technical interview question about {skill}."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a technical interviewer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error generating GPT question for {skill}: {e}"

# === Step 6: Sample HotpotQA Context ===
!pip install datasets
from datasets import load_dataset

hotpot_data = load_dataset("hotpot_qa", "fullwiki", split="train[:1000]")

def find_context_for_skill(skill, data, max_results=1):
    for item in data:
        for context_info in item.get("context", []):  # Iterate through context_info lists
            title_list = context_info[0]  # Get the title list (first element)
            paragraph = context_info[1]  # Get the paragraph (second element)

            if isinstance(paragraph, str) and skill.lower() in paragraph.lower():
                [].append(paragraph)
                if len([]) >= max_results:
                    return []
    return []

# === Step 7: Full Pipeline ===
print("\n\n--- 🚀 Interview Questions ---\n")
for skill in skills:
    print(f"🧠 Skill: {skill.capitalize()}")
    answer = get_answer_with_bert(f"What is {skill}?", hotpot_context)
    print("📘 Context-based Answer:", answer)
    question = generate_gpt_question(skill)
    print("📝 Generated Question:", question)
    print("\n" + "-" * 40 + "\n")
