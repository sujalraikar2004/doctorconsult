import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { FiMic, FiImage, FiPlay, FiPause, FiX, FiDownload, FiAlertCircle } from 'react-icons/fi';
import { Document, Packer, Paragraph, TextRun, HeadingLevel } from 'docx';
import { saveAs } from 'file-saver';

export default function App() {
  // State management
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState('English');
  const [allergies, setAllergies] = useState('');
  const [recommendation, setRecommendation] = useState(null);
  const [imageAnalysis, setImageAnalysis] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState('');
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);

  // Generate Word Document
  const generateWordDocument = (recommendation) => {
    const doc = new Document({
      sections: [
        {
          properties: {},
          children: [
            new Paragraph({
              text: "Medical Diagnosis Report",
              heading: HeadingLevel.HEADING_1,
              spacing: { after: 200 },
            }),
            new Paragraph({
              children: [
                new TextRun({
                  text: "Patient Report:",
                  bold: true,
                }),
              ],
            }),
            new Paragraph(recommendation.recommendation.replace(/\*\*/g, '')),
            new Paragraph({
              text: "Recommended Specialists",
              heading: HeadingLevel.HEADING_2,
              spacing: { before: 400, after: 200 },
            }),
            ...recommendation.doctors.map((doctor) =>
              new Paragraph({
                children: [
                  new TextRun({
                    text: `${doctor.name} (${doctor.specialization})`,
                    bold: true,
                  }),
                  new TextRun({
                    text: `\nHospital: ${doctor.hospital}\nRating: ${doctor.rating}/5`,
                    break: 1,
                  }),
                  new TextRun({
                    text: doctor.medicines?.length > 0
                      ? `\nMedicines: ${doctor.medicines.join(', ')}`
                      : '\nNo medicines suggested',
                    break: 1,
                  }),
                ],
                spacing: { after: 150 },
              })
            ),
          ],
        },
      ],
    });

    Packer.toBlob(doc).then((blob) => {
      saveAs(blob, 'medical-report.docx');
    });
  };

  // Audio recording handlers
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      mediaRecorder.current.ondataavailable = (e) => {
        audioChunks.current.push(e.data);
      };
      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioURL(audioUrl);
        audioChunks.current = [];
      };
      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access required for recording');
    }
  };

  const stopRecording = () => {
    mediaRecorder.current.stop();
    setIsRecording(false);
  };

  const handleTextToSpeech = async () => {
    try {
      const res = await fetch('http://localhost:5000/text-to-speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: recommendation.recommendation,
          lang: LANGUAGE_MAP[language],
        }),
      });

      if (!res.ok) throw new Error('Failed to generate speech');

      const blob = await res.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      // Add error handling for audio playback
      audio.addEventListener('error', (e) => {
        console.error('Audio playback error:', e);
        setError('Failed to play audio');
      });

      audio.play();
    } catch (err) {
      console.error('Text-to-speech error:', err.message);
      setError(`Text-to-speech error: ${err.message}`);
    }
  };

  const handleRecommendation = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language,
          allergies,
          question: query,
        }),
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();

      if (data.error) throw new Error(data.error);
      setRecommendation(data);
      setImageAnalysis('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const LANGUAGE_MAP = {
    English: 'en',
    Hindi: 'hi',
    Marathi: 'mr',
    Kannada: 'kn',
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-cyan-50 p-4 sm:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-800 bg-clip-text bg-gradient-to-r from-teal-600 to-cyan-500">
            MediCare AI Assistant
          </h1>
          <p className="text-gray-600">Advanced Medical Diagnosis Platform</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-2xl shadow-xl p-6 space-y-6 border border-teal-100">
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Describe Your Symptoms
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full px-4 py-3 rounded-lg border border-teal-200 focus:border-teal-400 focus:ring-2 focus:ring-teal-100 transition-all resize-none"
                  rows="4"
                  placeholder="Enter your medical query or record audio below..."
                />
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Voice Recording
                </label>
                <div className="flex items-center gap-4">
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                      isRecording
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : 'bg-teal-500 hover:bg-teal-600 text-white'
                    }`}
                  >
                    {isRecording ? (
                      <>
                        <FiPause className="text-lg" />
                        Stop Recording
                      </>
                    ) : (
                      <>
                        <FiMic className="text-lg" />
                        Start Recording
                      </>
                    )}
                  </button>
                  {audioURL && (
                    <audio controls className="flex-1">
                      <source src={audioURL} type="audio/wav" />
                    </audio>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">Language</label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full pl-3 pr-8 py-2.5 rounded-lg border border-teal-200 focus:border-teal-400 focus:ring-2 focus:ring-teal-100"
                  >
                    {Object.keys(LANGUAGE_MAP).map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">Allergies</label>
                  <input
                    type="text"
                    value={allergies}
                    onChange={(e) => setAllergies(e.target.value)}
                    placeholder="e.g., Penicillin, Nuts"
                    className="w-full px-3 py-2.5 rounded-lg border border-teal-200 focus:border-teal-400 focus:ring-2 focus:ring-teal-100"
                  />
                </div>
              </div>

              <button
                onClick={handleRecommendation}
                disabled={loading}
                className="w-full h-12 flex items-center justify-center gap-2 bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg transition-all disabled:opacity-50 disabled:pointer-events-none"
              >
                {loading ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Analyzing...
                  </div>
                ) : (
                  <>
                    <FiPlay className="text-lg" />
                    Get Diagnosis
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-6 space-y-6 border border-teal-100">
            {error && (
              <div className="p-4 bg-red-50 border border-red-100 rounded-xl flex items-start gap-3 text-red-700">
                <FiAlertCircle className="flex-shrink-0 mt-1" />
                <div>{error}</div>
              </div>
            )}

            {recommendation && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-semibold text-gray-800">Diagnosis Report</h2>
                  <button
                    onClick={() => generateWordDocument(recommendation)}
                    className="flex items-center gap-2 px-4 py-2 bg-teal-500 hover:bg-teal-600 text-white rounded-lg transition-colors"
                  >
                    <FiDownload className="text-lg" />
                    Download Report
                  </button>
                </div>
                <div className="prose max-w-none text-gray-700">
                  <ReactMarkdown>{recommendation.recommendation}</ReactMarkdown>
                </div>

                {recommendation.doctors?.length > 0 && (
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800">Recommended Specialists</h3>
                    <div className="grid gap-4">
                      {recommendation.doctors.map((doctor, index) => (
                        <div
                          key={index}
                          className="p-4 bg-teal-50 rounded-xl border border-teal-100"
                        >
                          <div className="flex items-center gap-4">
                            <div className="flex-1">
                              <h4 className="font-medium text-gray-800">{doctor.name}</h4>
                              <div className="text-sm text-gray-600 mt-1 space-y-1">
                                <p>
                                  <span className="font-medium">Specialization:</span>{' '}
                                  {doctor.specialization}
                                </p>
                                <p>
                                  <span className="font-medium">Hospital:</span> {doctor.hospital}
                                </p>
                                <p>
                                  <span className="font-medium">Rating:</span>{' '}
                                  <span className="text-teal-600">{doctor.rating}/5</span>
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {imageAnalysis && (
              <div className="space-y-4">
                <h2 className="text-2xl font-semibold text-gray-800">Imaging Analysis</h2>
                <div className="prose max-w-none text-gray-700">
                  <ReactMarkdown>{imageAnalysis}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}