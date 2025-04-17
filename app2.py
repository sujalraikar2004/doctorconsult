from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import time
import base64
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
import whisper
from langchain.docstore.document import Document
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
model = whisper.load_model("base")

LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Kannada": "kn"
}

# Load data and initialize vectors
def initialize_system():
    data =[
    {
      "id": "01",
      "name": "Dr. Alfaz Ahmed",
      "specialization": "Surgeon",
      "avgRating": 4.9,
      "totalRating": 272,
      "totalPatients": 1200,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "Consult a surgeon if you have symptoms like severe abdominal pain, broken bones, deep cuts, or need surgical intervention for any organ-related issues.",
      "medicines": ["Ibuprofen", "Paracetamol", "Morphine", "Antibiotics (Amoxicillin)", "Lidocaine (for local anesthesia)"]
    },
    {
      "id": "02",
      "name": "Dr. Saleh Mahmud",
      "specialization": "Neurologist",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 1678,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "Visit a neurologist if you experience symptoms like frequent headaches, dizziness, blurred vision, memory problems, numbness, or seizures.",
      "medicines": ["Diazepam", "Topiramate", "Gabapentin", "Carbamazepine", "Levetiracetam"]
    },
    {
      "id": "03",
      "name": "Dr. Farid Uddin",
      "specialization": "Dermatologist",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 986,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "A dermatologist should be consulted for skin-related concerns such as rashes, acne, eczema, psoriasis, or if you notice unusual moles or skin changes.",
      "medicines": ["Tretinoin", "Hydrocortisone", "Clindamycin", "Benzoyl Peroxide", "Antihistamines (Cetirizine)"]
    },
    {
      "id": "04",
      "name": "Dr. Vikram Singhania",
      "specialization": "Cardiologist",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 1803,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "Consult a cardiologist if you have symptoms like chest pain, shortness of breath, irregular heartbeat, swelling in legs, or fatigue during physical activity.",
      "medicines": ["Aspirin", "Atorvastatin", "Beta-blockers (Metoprolol)", "ACE inhibitors (Lisinopril)", "Nitroglycerin"]
    },
    {
      "id": "05",
      "name": "Dr. Anand Khanna",
      "specialization": "Psychiatrist",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 1298,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "A psychiatrist should be consulted if you're experiencing symptoms like anxiety, depression, mood swings, excessive worry, or mental health issues affecting daily life.",
      "medicines": ["Sertraline", "Fluoxetine", "Lorazepam", "Citalopram", "Quetiapine"]
    },
    {
      "id": "06",
      "name": "Dr. Kavita Gupta",
      "specialization": "Ophthalmologist",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 1291,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "Consult an ophthalmologist if you have symptoms such as eye pain, blurred vision, double vision, excessive tearing, or difficulty seeing clearly.",
      "medicines": ["Latanoprost (for glaucoma)", "Artificial tears", "Prednisolone acetate", "Tobramycin (antibiotic eye drops)", "Cyclopentolate (for pupil dilation)"]
    },
    {
      "id": "07",
      "name": "Dr. Priya Patel",
      "specialization": "Pediatrician",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 1382,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "Visit a pediatrician if your child experiences fever, difficulty breathing, developmental delays, persistent coughing, or any issues related to their growth and development.",
      "medicines": ["Paracetamol", "Amoxicillin", "Diphenhydramine", "Salbutamol (for asthma)", "Ibuprofen"]
    },
    {
      "id": "08",
      "name": "Dr. Suman Verma",
      "specialization": "Endocrinologist",
      "avgRating": 4.8,
      "totalRating": 272,
      "totalPatients": 784,
      "hospital": "Mount Adora Hospital, Sylhet.",
      "prescription": "An endocrinologist should be consulted for symptoms such as weight gain or loss, unexplained fatigue, changes in appetite, or issues related to thyroid, diabetes, or hormonal imbalances.",
      "medicines": ["Levothyroxine (for hypothyroidism)", "Metformin (for diabetes)", "Insulin", "Prednisone (for adrenal insufficiency)", "Fludrocortisone"]
    },
    {
      "id": "09",
      "name": "Dr. Nisha Mehta",
      "specialization": "Gynecologist",
      "avgRating": 4.7,
      "totalRating": 200,
      "totalPatients": 1500,
      "hospital": "Green Valley Hospital, Dhaka.",
      "prescription": "Consult a gynecologist if you have symptoms like irregular periods, pelvic pain, abnormal bleeding, or reproductive health concerns.",
      "medicines": ["Contraceptive pills", "Progesterone", "Metformin (for PCOS)", "Paracetamol", "Hormone replacement therapy"]
    },
    {
      "id": "10",
      "name": "Dr. Arun Kumar",
      "specialization": "Orthopedic",
      "avgRating": 4.9,
      "totalRating": 310,
      "totalPatients": 2200,
      "hospital": "Orthopedic Care Center, Dhaka.",
      "prescription": "Visit an orthopedic specialist if you experience symptoms such as joint pain, fractures, arthritis, or any musculoskeletal injuries.",
      "medicines": ["Ibuprofen", "Paracetamol", "Methotrexate (for arthritis)", "Calcium and Vitamin D", "Tramadol"]
    },
    {
      "id": "11",
      "name": "Dr. Ramesh Sharma",
      "specialization": "Gastroenterologist",
      "avgRating": 4.6,
      "totalRating": 180,
      "totalPatients": 900,
      "hospital": "City Medical Hospital, Dhaka.",
      "prescription": "Consult a gastroenterologist for symptoms like chronic stomach pain, indigestion, acid reflux, nausea, or abnormal bowel movements.",
      "medicines": ["Pantoprazole", "Loperamide", "Metoclopramide", "Rifaximin", "Antacids"]
    },
    {
      "id": "12",
      "name": "Dr. Tanvi Sharma",
      "specialization": "Radiologist",
      "avgRating": 4.7,
      "totalRating": 260,
      "totalPatients": 1400,
      "hospital": "Radiology Imaging Center, Dhaka.",
      "prescription": "Consult a radiologist for imaging services such as X-rays, CT scans, MRIs, or ultrasound for diagnostic purposes related to various health issues.",
      "medicines": []
    },
    {
      "id": "13",
      "name": "Dr. Rajesh Kumar",
      "specialization": "Urologist",
      "avgRating": 4.8,
      "totalRating": 250,
      "totalPatients": 1100,
      "hospital": "Metro Urology Clinic, Dhaka.",
      "prescription": "Visit a urologist for symptoms like urinary tract infections, kidney stones, frequent urination, or issues related to the bladder or prostate.",
      "medicines": ["Tamsulosin", "Finasteride", "Ciprofloxacin", "Amoxicillin", "Ibuprofen"]
    }
]
    
    documents = [
        Document(
            page_content=item["prescription"],
            metadata={
                "name": item["name"],
                "specialization": item["specialization"],
                "hospital": item["hospital"],
                "medicines": ", ".join(item["medicines"])
            }
        ) for item in data
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_documents(split_documents, embeddings)
    return vectors, data

vectors, doctors_data = initialize_system()

def get_prompt_template():
    return ChatPromptTemplate.from_template("""
    You are a professional medical assistant. Based on the provided context, identify the most relevant doctor for the given symptoms. 
    Provide a medical assessment, suggested medicines (excluding allergies), and home remedies.
    Respond in {language} language. Use simple, patient-friendly language.

    Patient Allergies: {allergies}

    Context:
    {context}

    Symptoms: {input}

    Answer in markdown format with:
    - **Doctor Recommendation**
    - **Medical Assessment**
    - **Suggested Medicines** (state if none due to allergies)
    - **Home Remedies**
    """)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        language = data.get('language', 'English')
        allergies = data.get('allergies', '')
        question = data.get('question')
        audio = data.get('audio')

        if audio:
            try:
                audio_bytes = base64.b64decode(audio)
                question = transcribe_audio(audio_bytes)
            except Exception as e:
                return jsonify({"error": f"Audio processing failed: {str(e)}"}), 400

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Process request
        prompt = get_prompt_template()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            'input': question,
            'language': language,
            'allergies': allergies
        })

        result = process_response(response, language, allergies)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_response(response, language, allergies):
    answer = response['answer']
    context = response['context']
    
    doctors = []
    if context:
        try:
            top_specialization = context[0].metadata['specialization']
            matched_doctors = [
                doc for doc in doctors_data
                if doc['specialization'] == top_specialization
            ]
            
            allergy_list = [a.strip().lower() for a in allergies.split(',') if a.strip()]
            
            for doctor in matched_doctors:
                filtered_meds = [
                    med for med in doctor['medicines']
                    if med.strip().lower() not in allergy_list
                ] if allergy_list else doctor['medicines']
                
                doctors.append({
                    "name": doctor['name'],
                    "specialization": doctor['specialization'],
                    "hospital": doctor['hospital'],
                    "rating": doctor['avgRating'],
                    "patients": doctor['totalPatients'],
                    "medicines": filtered_meds,
                    "prescription": doctor['prescription']
                })
        except Exception as e:
            app.logger.error(f"Doctor processing error: {str(e)}")

    return {
        "recommendation": answer,
        "doctors": doctors,
        "context": [str(c.page_content) for c in context]
    }

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        image_file = request.files['image']
        query = request.form.get('query', '')
        language = request.form.get('language', 'English')

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            image_file.save(tmp_file.name)
            encoded_image = base64.b64encode(open(tmp_file.name, "rb").read()).decode('utf-8')

        client = Groq(api_key=groq_api_key)
        messages = [{
            "role": "user", 
            "content": [
                {"type": "text", "text": f"{query} Respond in {language} language"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        
        analysis_result = response.choices[0].message.content
        return jsonify({"analysis": analysis_result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def transcribe_audio(audio_bytes):
    try:
        wav_audio = BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(wav_audio, format="wav")
        mp3_buffer = BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3") as fp:
            fp.write(mp3_buffer.read())
            result = model.transcribe(fp.name)
        return result["text"]
    except Exception as e:
        raise Exception(f"Audio processing error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)