import streamlit as st
import os
import uuid
import time
import base64
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from gtts import gTTS
import streamlit_mic_recorder as mic_recorder
from pydub import AudioSegment
from io import BytesIO
import whisper
from langchain.docstore.document import Document
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Language configuration
LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Kannada": "kn"
}

# Configure Streamlit app
st.set_page_config(page_title="Medical Specialist Recommender", page_icon="🏥", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton>button {
        background-color: #008CBA;
        color: white;
    }
    .warning {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'allergies' not in st.session_state:
    st.session_state.allergies = ""

# LLM setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Image processing functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(query, image_path, language):
    try:
        encoded_image = encode_image(image_path)
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
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not analyze image: {str(e)}"

# Updated prompt template with allergy checking
def get_prompt_template():
    return ChatPromptTemplate.from_template("""
You are a professional medical assistant. Based on the provided context, identify the most relevant doctor for the given symptoms. 
Provide a medical assessment, suggested medicines prescribed by the doctor (excluding any the patient is allergic to), and any applicable home remedies related to the condition. 
Respond in {language} language. Use simple, patient-friendly language while maintaining medical accuracy.

Patient Allergies: {allergies}

Context:
{context}

Symptoms: {input}

Answer in markdown format with:
- **Doctor Recommendation**
- **Medical Assessment**
- **Suggested Medicines** (clearly state if none due to allergies)
- **Home Remedies**
""")

# Load and process static JSON data
def load_and_process_json():
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

    
    st.session_state.doctors_data = data
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
    st.session_state.vectors = FAISS.from_documents(split_documents, embeddings)
    st.session_state.processed = True

# Initial data processing
if not st.session_state.processed:
    with st.spinner("Loading and processing medical data..."):
        load_and_process_json()

# Voice functions
def speak_text(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    tts.save(filename)
    with open(filename, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")
    os.remove(filename)

def transcribe_audio(audio_bytes):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(suffix=".mp3") as fp:
        fp.write(audio_bytes)
        result = model.transcribe(fp.name)
    return result["text"]

# Main UI
st.title("🏥 Medical Specialist Recommender")
st.markdown("Describe your symptoms via voice or text to get specialist recommendations.")

with st.sidebar:
    selected_language = st.selectbox("Select Language", list(LANGUAGE_MAP.keys()))
    lang_code = LANGUAGE_MAP[selected_language]
    
    # Allergy input
    st.session_state.allergies = st.text_input(
        "List any medicine allergies (comma-separated):",
        help="Example: Penicillin, Ibuprofen, Aspirin"
    )
    
    # Image uploader
    uploaded_image = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])

# Process image if uploaded
image_analysis = None
if uploaded_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_image.getvalue())
        image_path = tmp_file.name
    
    image_query = st.text_input("Enter your question about the image:", key="image_query")
    if image_query:
        image_analysis = analyze_image(image_query, image_path, selected_language)
        st.markdown("### 🖼️ Image Analysis")
        st.markdown(image_analysis)
        speak_text(image_analysis, lang_code)

# Audio + Text input
audio = mic_recorder.mic_recorder(
    start_prompt="🎙️ Click to record symptoms",
    stop_prompt="Recording... click to stop",
    use_container_width=True,
    format="wav",
    key="recorder"
)
typed_question = st.text_input("Or type your symptoms here:")

# Decide final question
question = None
if audio and 'bytes' in audio:
    try:
        wav_audio = BytesIO(audio['bytes'])
        audio_segment = AudioSegment.from_file(wav_audio, format="wav")
        mp3_buffer = BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)
        question = transcribe_audio(mp3_buffer.read())
        st.success(f"Transcribed symptoms: {question}")
    except Exception as e:
        st.warning("Could not process audio recording.")
        st.exception(e)

if not question and typed_question:
    question = typed_question

# Process question
if question and st.session_state.processed:
    start_time = time.time()
    
    # Get allergy information
    allergies = st.session_state.allergies.strip() or "No known allergies"
    
    # Create chain
    prompt = get_prompt_template()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Invoke chain with allergy information
    response = retrieval_chain.invoke({
        'input': question,
        'language': selected_language,
        'allergies': allergies
    })
    
    processing_time = time.time() - start_time
    
    # Display answer
    st.markdown("### 🩺 Recommendation")
    st.markdown(response['answer'])
    speak_text(response['answer'], lang_code)

    # Display detailed doctor information with allergy filtering
    if response['context']:
        try:
            top_specialization = response['context'][0].metadata['specialization']
            matched_doctors = [
                doc for doc in st.session_state.doctors_data
                if doc['specialization'] == top_specialization
            ]
            
            if matched_doctors:
                st.markdown("### 👨⚕️ Recommended Specialist Details")
                
                # Process allergies for filtering
                allergy_list = [a.strip().lower() for a in st.session_state.allergies.split(',') if a.strip()]
                
                for doctor in matched_doctors:
                    # Filter medicines
                    if allergy_list:
                        filtered_meds = [
                            med for med in doctor['medicines']
                            if med.strip().lower() not in allergy_list
                        ]
                    else:
                        filtered_meds = doctor['medicines']
                    
                    # Create columns for layout
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.subheader(doctor['name'])
                        st.write(f"**Specialization:** {doctor['specialization']}")
                        st.write(f"🏥 **Hospital:** {doctor['hospital']}")
                        st.write(f"⭐ **Rating:** {doctor['avgRating']} ({doctor['totalRating']} reviews)")
                        st.write(f"👥 **Patients Treated:** {doctor['totalPatients']}")
                    
                    with col2:
                        st.markdown("**Approved Medicines:**")
                        if filtered_meds:
                            st.write(", ".join(filtered_meds))
                        else:
                            st.markdown("<div class='warning'>No safe medications available due to allergies</div>", 
                                      unsafe_allow_html=True)
                        
                        st.markdown("**Typical Prescription Guidelines:**")
                        st.info(doctor['prescription'])
                    
                    st.markdown("---")
        except Exception as e:
            st.warning("Could not retrieve doctor details")

    # Show reference sections
    with st.expander("📄 View matching specialist profiles"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"#### Specialist {i+1}")
            st.write(f"**Name:** {doc.metadata['name']}")
            st.write(f"**Specialization:** {doc.metadata['specialization']}")
            st.write(f"**Hospital:** {doc.metadata['hospital']}")
            st.markdown("---")

    st.caption(f"Processed in {processing_time:.2f} seconds")

elif question and not st.session_state.processed:
    st.warning("Data is still processing. Please wait...")
elif not question:
    st.info("Please describe your symptoms using voice or text.")

# Footer
st.markdown("---")
st.markdown("""
💡 **Example Symptoms**:
- "I have severe chest pain and shortness of breath"
- "Experiencing persistent headaches and dizziness"
- "Skin rash with itching and redness"
- "Frequent urination and abdominal pain"
- "Blurred vision and eye pain"
""")