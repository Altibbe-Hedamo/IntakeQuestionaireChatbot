from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from psycopg2 import sql
import psycopg2
import json
import os
import base64
import mimetypes
import tempfile
from docx import Document
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import pytesseract
import fitz
import io
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)

from dotenv import load_dotenv

load_dotenv()

limiter = Limiter(
    key_func=get_remote_address,
)
limiter.init_app(app)

import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

API_KEY = os.getenv("OCR_SPACE_API_KEY")
API_URL = "https://api.ocr.space/parse/image"

DB_URL=os.getenv("DB_URL")
NEON_DB_URL=os.getenv("NEON_DB_URL")
DB_URL_FOR_REPORT=os.getenv("DB_URL_FOR_REPORT")

def get_db_connection():
    return psycopg2.connect(DB_URL)

def get_db_connection1():
    return psycopg2.connect(NEON_DB_URL)

def get_db_connection2():
    return psycopg2.connect(DB_URL_FOR_REPORT)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            history TEXT,
            answers TEXT,
            completed BOOLEAN,
            attempts TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

initial_question = "What is your name of company?"

def query_data(conn, category, subcategory):
    cursor = conn.cursor()
    try:
        cursor.execute(
            sql.SQL("SELECT content FROM {} WHERE subcategory_name = %s;")
            .format(sql.Identifier(category)),
            (subcategory,)
        )
        results = cursor.fetchall()
        return [row[0] for row in results] 
    except Exception as e:
        return []
    finally:
        cursor.close()

def query_data_for_report(conn, table, subcategory):
    cursor = conn.cursor()
    try:
        cursor.execute(
            sql.SQL("SELECT content FROM {} WHERE subcategory_name = %s;")
            .format(sql.Identifier(table)),
            (subcategory,)
        )
        results = cursor.fetchall()
        return [row[0] for row in results] 
    except Exception as e:
        return []
    finally:
        cursor.close()

def create_product_table():
    conn = get_db_connection()
    if conn is None:
        return "Database connection error."

    try:
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS product_info (
            id SERIAL PRIMARY KEY,
            product_name TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
        return "Table 'product_info' created successfully (if not already existing)."
    except Exception as e:
        return f"Error creating table: {e}"

create_product_table()

def insert_product(product_name, content):
    conn = get_db_connection()
    if conn is None:
        return "Database connection error."

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO product_info (product_name, content)
        VALUES (%s, %s)
        ON CONFLICT (product_name) DO UPDATE
        SET content = product_info.content || '\n' || EXCLUDED.content
        """
        cursor.execute(query, (product_name, content))
        conn.commit()

        cursor.close()
        conn.close()

        return "Product content updated successfully."
    except Exception as e:
        return f"Error: {e}"

def get_product_content(product_name):
    conn = get_db_connection()
    if conn is None:
        return "Database connection error."

    try:
        cursor = conn.cursor()
        query = "SELECT content FROM product_info WHERE product_name = %s"
        cursor.execute(query, (product_name,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return result[0] if result else "Product not found."
    except Exception as e:
        return f"Error: {e}"

def get_few_shot_prompt(conn, table, subcategory):
    questions_list = query_data(conn, table, subcategory)
    questions_text = questions_list[0] if questions_list else ""

    shots = "Example questions you learn while asking Questions:\n"
    shots += questions_text.strip() + "\n\n"
    print("Shots prompt::->",shots)
    return shots

'''def ask_model(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error from Gemini:", e)
        return "Sorry, I couldn't generate a follow-up question."
        '''

'''def generate_next_question(history):
    few_shot = get_few_shot_prompt()
    history_text = "\n".join(history[-10:])
    input_text = (
        f"{few_shot} "
        f"You are an AI assistant conducting a structured interview to gather transparency data for a product.\n"
        f"Here is the conversation so far:\n{history_text}\n\n"
        f"Ask the next logical question that builds upon previous answers. Avoid repetition."
    )
    next_question = ask_model(input_text)
    print("AI Response:", next_question)
    return next_question'''

def get_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT history, answers, completed, attempts FROM user_sessions WHERE session_id = %s", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        history = json.loads(row[0])
        answers = json.loads(row[1])
        completed = row[2]
        attempts = json.loads(row[3]) if row[3] else {}
    else:
        history = []
        answers = []
        completed = False
        attempts = {}
        save_session(session_id, history, answers, completed, attempts)

    return history, answers, completed, attempts


def save_session(session_id, history, answers, completed, attempts):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_sessions (session_id, history, answers, completed, attempts)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (session_id)
        DO UPDATE SET 
            history = EXCLUDED.history, 
            answers = EXCLUDED.answers, 
            completed = EXCLUDED.completed,
            attempts = EXCLUDED.attempts
    ''', (
        session_id,
        json.dumps(history),
        json.dumps(answers),
        completed,
        json.dumps(attempts)
    ))
    conn.commit()
    conn.close()

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def ocr_space_file(image_bytes, language="eng", filetype="JPG"):
    """
    Sends an image file (in bytes) to the OCR API and returns the extracted text.
    """
    payload = {
        "apikey": API_KEY,
        "language": language,
        "isOverlayRequired": False,
        "filetype": filetype
    }
    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            data=payload,
        )
        result = response.json()

        print("OCR API Response:", result)  

        if result.get("IsErroredOnProcessing", False):
            return {"error": result.get("ErrorMessage", "Unknown error")}

        if "ParsedResults" not in result or not result["ParsedResults"]:
            return {"error": "No text extracted or invalid response format"}

        return {"text": result["ParsedResults"][0].get("ParsedText", "")}

    except Exception as e:
        return {"error": str(e)}


def extract_text(file_path,language,use_openocr=True):
    """
    Extracts text from a given file:
    - PDF: Converts pages to images using PyMuPDF, then uses OpenOCR or Tesseract.
    - DOCX: Uses `python-docx` to extract text.
    - Images: Uses OpenOCR or Tesseract OCR.
    - TXT: Reads the plain text.
    """
    text = ""
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap()
                img_byte_arr = io.BytesIO()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(img_byte_arr, format="JPEG", quality=80)
                img_bytes = img_byte_arr.getvalue()

                extracted_result = (
                    ocr_space_file(img_bytes, language, filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(img)}
                )

                if "error" in extracted_result:
                    return extracted_result 

                extracted_text = extracted_result["text"]
                print(extracted_text)
                if extracted_text:
                    text += f"Page {page_num + 1}:\n{extracted_text}\n\n"

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            with open(file_path, "rb") as img_file:
                img_bytes = img_file.read()
                extracted_result = (
                    ocr_space_file(img_bytes,language,filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(Image.open(file_path))}
                )

                if "error" in extracted_result:
                    return extracted_result  

                text = extracted_result["text"]

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

    except Exception as e:
        return {"error": str(e)}

    return text.strip()

@app.route('/chats', methods=['POST'])
def chat():
    contents = []
    session_id = None
    user_response = None
    temp_file_path = None

    try:
        if request.content_type.startswith('multipart/form-data'):
            session_id = request.form.get("session_id")
            user_response = request.form.get("user_response")
            file = request.files.get("file")

            if file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
                    file.save(temp_file.name)
                    temp_file_path = temp_file.name

                mime = get_mime_type(temp_file_path)

                if mime in ['application/pdf', 'image/jpeg', 'image/png', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    with open(temp_file_path, "rb") as f:
                        file_bytes = f.read()

                    extracted_result = extract_text(temp_file_path, language="eng", use_openocr=True)
                    print("Extracted text from File:")
                    print(extracted_result)

                    if isinstance(extracted_result, dict) and "error" in extracted_result:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"File processing error: {extracted_result['error']}"}), 400

                    if extracted_result.strip():
                        user_response = extracted_result.strip()

                    contents.append({
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(file_bytes).decode('utf-8')
                        }
                    })

                elif mime.startswith("audio/"):
                    try:
                        audio = AudioSegment.from_file(temp_file_path)
                        wav_path = temp_file_path + ".wav"
                        audio.export(wav_path, format="wav")

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio_data = recognizer.record(source)

                        user_response = recognizer.recognize_google(audio_data)
                        print("Transcribed text:", user_response)

                        os.unlink(wav_path)
                    except sr.UnknownValueError:
                        os.unlink(temp_file_path)
                        return jsonify({"error": "Could not understand audio."}), 400
                    except sr.RequestError as e:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"Speech recognition service error: {e}"}), 500
                    except Exception as e:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"Audio transcription failed: {str(e)}"}), 400
                else:
                    os.unlink(temp_file_path)
                    return jsonify({"error": f"Unsupported MIME type: {mime}. Supported types: PDF, JPEG, PNG, DOCX, TXT, AUDIO."}), 400

        else:
            data = request.get_json(silent=True)
            if not data:
               return jsonify({"error": "Missing or invalid JSON data"}), 400

            session_id = data.get("session_id")
            user_response = data.get("user_response")
            category=data.get("category","")
            sub_categories=data.get("sub_categories",[])
            product_name=data.get("product_name","")

            print("category::->",category)
            print("Subcategories::->",sub_categories)
            print("product_name::->",product_name)

        history, answers, completed, attempts = get_session(session_id)
        if attempts is None:
            attempts = {}

        if not history:
          conn = get_db_connection1()
          all_questions = ""

          for subcategory in sub_categories:
              prompt_text = get_few_shot_prompt(conn, category, subcategory)
              all_questions += prompt_text + "\n" 
              print("all questions in not in history::::->",all_questions)

          questions = [line.strip() for line in all_questions.split("\n") if line.strip()]

          print("Info from Database::->",questions)

          history.append(product_name)
          history.append(category)
          history.append(json.dumps(sub_categories)) 

          history.extend(questions)
          save_session(session_id, history, answers, completed, attempts)

          history.append(initial_question)
          save_session(session_id, history, answers, completed, attempts)

          return jsonify({
                 "message": initial_question,
                 "completed": False,
                 "question_number": 1
                  })
            
        if user_response or contents:
            print("User Response")
            print(user_response)
            last_question = history[-1]
            question_number = len(answers) + 1
            question_key = f"question_{question_number}"

            history_text = '\n'.join(history)
            
            validation_prompt = (
            f"You are an expert AI assistant to validate the User Responses. A user was asked the following question:\n"
            f"Q: {last_question}\n\n"
            f"They responded with:\nA: {user_response or '[File Uploaded]'}\n\n"
            f"Conversation history so far:\n{history_text}\n\n"
            f"Your task is to validate the user's response based on the following criteria:\n"
            f"1. The answer must be correct, complete, and factually accurate.\n"
            f"2. The answer must not contradict any previous responses in the conversation.\n"
            f"3. Do not mark responses as incorrect for generic or straightforward answers like product name, company name, brand, or place of production—unless they clearly contradict earlier responses or are obviously wrong.\n\n"
            f"Output 'Correct' if the response is accurate and meets all the above criteria.\n"
            f"Otherwise, output 'Incorrect' followed by a brief explanation of why the response is incorrect.\n"
            f"Do not include any explanation if the response is correct."
            )

            validation_result = model.generate_content(validation_prompt).text.strip().lower()
            print("validation result")
            print(validation_result)

            if validation_result == "correct":
                print("Validation was correct")
                answers.append(user_response or "[File Uploaded]")
                contents.insert(0, {"text": f"Q: {last_question}\nA: {user_response}"})
                history.append(f"A: {user_response}")
                attempts.pop(question_key, None)
                save_session(session_id, history, answers, completed, attempts)
            else:
                attempts[question_key] = attempts.get(question_key, 0) + 1
                print("Inside the Wrong answer and increasing the attempt")
                if attempts[question_key] >= 3:
                    answers.append(user_response)
                    contents.insert(0, {"text": f"Q: {last_question}\nA: {user_response}"})
                    history.append(f"A: {user_response}")
                    save_session(session_id, history, answers, completed, attempts)

                else:
                    save_session(session_id, history, answers, completed, attempts)
                    return jsonify({
                        "message": f"{validation_result}. Attempt {attempts[question_key]} of 3.Please try again.\n\n{last_question}",
                        "completed": False,
                        "question_number": len(history)
                    })

        if len(answers) >= 20:
            completed = True
            report_prompt = "Generate a report summarizing the following questions and answers and instructions:\n"
            '''with open('prompt.txt', 'r', encoding='utf-8') as file:
                content = file.read()'''
            
            product_name=history[0]
            category = history[1]
            sub_categories = json.loads(history[2]) if isinstance(history[2], str) else history[2]
            conn=get_db_connection2()
            
            content_to_append_for_final_report=""
        
            all_content=""
            
            for subcategory in sub_categories:
              result = query_data_for_report(conn, category, subcategory)
              all_content += result[0] + "\n" 
              print("all questions in not in history::::->",all_content)
            
            report_prompt += all_content
            print("Report prompt before history",report_prompt)
            
            for i, a in enumerate(history): 
                report_prompt += f"{i+1}:{a}\n"
                content_to_append_for_final_report+= f"{i+1} : {a}"

            insert_product(product_name, content_to_append_for_final_report)

            print("Report prompt after history",report_prompt)

            contents.insert(0, {"text": report_prompt})
            response = model.generate_content(contents)
            report = response.text.strip()

            insert_product(product_name, report)

            content_from_db=get_product_content(product_name)
            print("Content Saved to DB for final report::::->>>>",content_from_db)

            save_session(session_id, history, answers, completed, attempts)
            return jsonify({
                "message": "Thank you! All questions answered.",
                "completed": True,
                "answers": answers,
                "report": report
            })

        joined_history = "\n".join(history)
        
        prompt = (
        f"You are a friendly and insightful AI assistant conducting a structured interview to gather transparency data for a product.\n"
        f"Here is the conversation so far:\n{joined_history}\n\n"
        f"Based on the responses provided so far, ask the next most relevant and logical question that builds on the previous answers.\n"
        f"Make sure the question is clear, concise, and naturally follows from the conversation. Avoid asking repetitive or redundant questions."
        )

        contents.insert(0, {"text": f"{prompt}\nWhat is the next appropriate question to ask?"})

        model_response = model.generate_content(contents)
        next_question = model_response.text.strip()

        history.append(next_question)
        save_session(session_id, history, answers, completed, attempts)

        return jsonify({
            "message": next_question,
            "completed": False,
            "question_number": len(history)
        })

    except Exception as e:
        print("Error in /chats:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file: {e}")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
