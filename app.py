import os
import time
import base64
import uuid
import json
import sqlite3
import datetime
import requests
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from dotenv import load_dotenv
from openai import OpenAI

# Azure Speech SDK is optional – the REST API is used instead for Render compatibility
try:
    import azure.cognitiveservices.speech as speechsdk
    _SPEECH_SDK_AVAILABLE = True
except Exception:
    speechsdk = None
    _SPEECH_SDK_AVAILABLE = False

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB max upload

# Ensure uploads folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), "uploads"), exist_ok=True)

# Database Initialization
DB_PATH = os.path.join(os.path.dirname(__file__), "support_tickets.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            customer_name TEXT,
            product_model TEXT,
            category TEXT,
            priority TEXT,
            sentiment TEXT,
            summary TEXT,
            details TEXT,
            status TEXT DEFAULT 'Open',
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Azure OpenAI / Foundry Configuration
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
TEXT_MODEL = os.getenv("TEXT_MODEL_DEPLOYMENT", "gpt-4.1-mini")
VISION_MODEL = os.getenv("VISION_MODEL_DEPLOYMENT") or TEXT_MODEL
IMAGE_MODEL = os.getenv("IMAGE_MODEL_DEPLOYMENT", "FLUX-1.1-pro")

# Azure Speech Service Configuration
SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT", "").rstrip("/")
SPEECH_KEY = os.getenv("SPEECH_API_KEY", "")
SPEECH_REGION = os.getenv("SPEECH_REGION", "eastus")

# Azure Content Understanding Configuration
CONTENT_ENDPOINT = os.getenv("CONTENT_ENDPOINT", "").rstrip("/")
CONTENT_KEY = os.getenv("CONTENT_API_KEY", "")
CONTENT_API_VERSION = os.getenv("CONTENT_API_VERSION", "2025-11-01")


def get_openai_client():
    if not AOAI_ENDPOINT or not AOAI_KEY:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be configured in .env")
    endpoint = AOAI_ENDPOINT.removesuffix("/openai/v1").rstrip("/")
    return OpenAI(api_key=AOAI_KEY, base_url=f"{endpoint}/openai/v1/")


def text_response(prompt, system="You are an expert product support assistant."):
    client = get_openai_client()
    response = client.responses.create(
        model=TEXT_MODEL,
        instructions=system,
        input=prompt
    )
    return response.output_text


# ==========================================
# ROUTES & CONTROLLERS
# ==========================================

@app.get("/")
def home():
    """Dashboard / landing page for Product Support Assistant."""
    return render_template("home.html")


# 1. Customer Support Chatbot
@app.get("/chat")
def chat():
    return render_template("chat.html")


@app.post("/api/chat")
def api_chat():
    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify(error="Please enter a customer question or inquiry."), 400

        system_instruction = (
            "You are 'ApexSupport AI', an empathetic, highly skilled, and professional senior product support specialist. "
            "Your goals are:\n"
            "1. Deliver clear, step-by-step troubleshooting instructions for hardware and software issues.\n"
            "2. Explain warranty policies, return/exchange criteria, and replacement steps accurately.\n"
            "3. Keep a warm, polite, reassuring, and solution-driven tone at all times.\n"
            "4. If an issue appears hazardous or cannot be solved remotely, guide the user on preparing for Tier-2 escalation."
        )

        formatted_input = []
        for h in history[-6:]:  # include up to last 6 conversational turns
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ["user", "assistant"] and content:
                formatted_input.append(f"{role.upper()}: {content}")
        
        formatted_input.append(f"USER: {message}")
        prompt = "\n".join(formatted_input)

        reply = text_response(prompt, system=system_instruction)
        return jsonify(reply=reply)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/chat/stream")
def api_chat_stream():
    """Real-time token streaming using Server-Sent Events (SSE)."""
    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify(error="Please enter a customer question or inquiry."), 400

        system_instruction = (
            "You are 'ApexSupport AI', an empathetic, highly skilled, and professional senior product support specialist. "
            "Deliver structured, step-by-step troubleshooting instructions using clean markdown formatting, lists, and bold callouts."
        )

        messages = [{"role": "system", "content": system_instruction}]
        for h in history[-6:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        client = get_openai_client()

        def generate():
            try:
                response = client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=messages,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield f"data: {json.dumps({'token': delta})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as ex:
                try:
                    full_text = text_response(message, system=system_instruction)
                    yield f"data: {json.dumps({'token': full_text})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as inner_ex:
                    yield f"data: {json.dumps({'error': str(inner_ex)})}\n\n"

        return Response(stream_with_context(generate()), content_type="text/event-stream")
    except Exception as e:
        return jsonify(error=str(e)), 500


# 2. Ticket & Feedback Sentiment / Text Analysis
@app.get("/ticket-analysis")
def ticket_analysis():
    return render_template("ticket_analysis.html")


@app.post("/api/ticket-analysis")
def api_ticket_analysis():
    try:
        data = request.json or {}
        ticket_text = data.get("text", "").strip()
        if not ticket_text:
            return jsonify(error="Please provide ticket text or customer feedback to analyze."), 400

        system_prompt = (
            "You are a Support Operations Intelligence Analyst. Analyze incoming support tickets and customer feedback "
            "with high precision to help support teams triage effectively."
        )

        prompt = f"""Analyze the following customer support ticket/review and return a structured assessment formatted with clear markdown headings and bullet points:

### 1. Issue Category & Subsystem
- Primary Category (e.g. Hardware Defect, Firmware/Software, Billing, Shipping, User Error)
- Affected Component/Model

### 2. Urgency & Priority Level
- Priority: [Critical | High | Medium | Low]
- SLA Recommendation (e.g. Respond within 1 hour, 4 hours, 24 hours)

### 3. Customer Sentiment & Frustration Meter
- Sentiment: [Angry | Highly Frustrated | Neutral | Inquisitive | Satisfied]
- Churn Risk: [High | Medium | Low]

### 4. Extracted Entities & Keywords
- Product Name / Model / Serial references:
- Error Codes / Symptoms:
- Key phrases:

### 5. Root Cause Hypothesis
- Potential cause of the malfunction or dissatisfaction

### 6. Recommended Support Action Plan & Draft Response
- Recommended internal action for Support Rep
- Ready-to-send drafted customer reply (empathetic, professional, actionable)

CUSTOMER TICKET TEXT:
{ticket_text}"""

        analysis_result = text_response(prompt, system=system_prompt)

        # Auto-persist to SQLite Ticket DB
        try:
            ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
            priority = "High" if "Critical" in analysis_result or "High" in analysis_result else "Medium"
            category = "Hardware/Firmware"
            summary_snippet = ticket_text[:140] + "..." if len(ticket_text) > 140 else ticket_text
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                INSERT INTO tickets (ticket_id, category, priority, sentiment, summary, details, source, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticket_id, category, priority, "Analyzed", summary_snippet, analysis_result, "Ticket Triage", "Open"))
            conn.commit()
            conn.close()
        except Exception:
            ticket_id = None

        return jsonify(result=analysis_result, saved_ticket_id=ticket_id)
    except Exception as e:
        return jsonify(error=str(e)), 500


# 3. Visual Product Troubleshooting & Defect Inspection
@app.get("/visual-troubleshoot")
def visual_troubleshoot():
    return render_template("visual_troubleshoot.html")


@app.post("/api/visual-troubleshoot")
def api_visual_troubleshoot():
    try:
        file = request.files.get("image")
        prompt = request.form.get("prompt", "Inspect this product image, detect physical damage or error status, and provide troubleshooting steps.").strip()
        
        if not file:
            return jsonify(error="Please upload a product photo, diagram, or error screen."), 400

        image_bytes = file.read()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        mimetype = file.mimetype or "image/jpeg"

        client = get_openai_client()
        system_instruction = (
            "You are an expert Hardware Quality Engineer and Technical Diagnostics Specialist. "
            "Analyze photos of products, broken components, error codes, port damage, LED patterns, and screen glitches. "
            "Provide accurate diagnostics, safety warnings, and step-by-step fix recommendations."
        )

        full_prompt = f"""Examine this product image carefully. Provide a structured diagnostic report:

### 1. Visual Inspection & Detected Anomalies
- Physical condition (cracks, burns, misalignments, loose connectors, water damage indicators, LED status, display error codes).

### 2. Diagnostic Assessment
- Likely Failure Mode / Problem diagnosis.
- Severity Rating: [Critical/Hazardous | Moderate Repairable | Minor/Cosmetic | Configuration Issue]

### 3. Safety Warning & Precautions
- Any electrical, battery, or mechanical safety hazards to be aware of.

### 4. Step-by-Step Resolution / Action Plan
- Exact troubleshooting or repair steps.
- Recommended replacement parts or RMA/Warranty return eligibility.

User Note / Symptom: {prompt}"""

        response = client.responses.create(
            model=VISION_MODEL,
            instructions=system_instruction,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": full_prompt},
                    {"type": "input_image", "image_url": f"data:{mimetype};base64,{b64_data}"}
                ]
            }]
        )
        return jsonify(result=response.output_text)
    except Exception as e:
        return jsonify(error=str(e)), 500


# 4. Voice-to-Text Support Call / Voicemail Transcriber
@app.get("/voice-support")
def voice_support():
    return render_template("voice_support.html")


@app.post("/api/voice-support")
def api_voice_support():
    """Transcribe audio using Azure Speech-to-Text REST API (no native SDK required)."""
    temp_path = None
    try:
        file = request.files.get("audio")
        if not file:
            return jsonify(error="Please upload or record an audio file (.wav, .mp3, .m4a, .ogg)."), 400

        if not SPEECH_KEY:
            return jsonify(error="Azure Speech Service key (SPEECH_API_KEY) is not configured in .env."), 400

        language = request.form.get("language", "en-US")

        # Save uploaded audio to a temp file
        ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
        temp_name = f"temp_audio_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(os.path.dirname(__file__), "uploads", temp_name)
        file.save(temp_path)

        # --- Azure Speech-to-Text via REST API ---
        # The STT REST endpoint is always region-based: https://<region>.stt.speech.microsoft.com
        # SPEECH_ENDPOINT in .env may be a generic Cognitive Services URL — don't use it as the STT base.
        # Only use SPEECH_ENDPOINT if it already points to the correct speech STT host.
        if SPEECH_ENDPOINT and "stt.speech.microsoft.com" in SPEECH_ENDPOINT:
            stt_base = SPEECH_ENDPOINT.rstrip("/")
        else:
            # Always derive from region — this is the correct Azure STT REST endpoint
            stt_base = f"https://{SPEECH_REGION}.stt.speech.microsoft.com"

        stt_url = (
            f"{stt_base}/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={language}&format=detailed"
        )

        # Determine content-type based on file extension
        content_type_map = {
            ".wav": "audio/wav; codecs=audio/pcm; samplerate=16000",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg; codecs=opus",
            ".m4a": "audio/aac",
            ".flac": "audio/flac",
        }
        content_type = content_type_map.get(ext, "audio/wav; codecs=audio/pcm; samplerate=16000")

        headers = {
            "Ocp-Apim-Subscription-Key": SPEECH_KEY,
            "Content-Type": content_type,
            "Accept": "application/json",
        }

        with open(temp_path, "rb") as audio_file:
            stt_response = requests.post(stt_url, headers=headers, data=audio_file, timeout=60)

        full_transcript = ""
        if stt_response.ok:
            stt_data = stt_response.json()
            # 'RecognitionStatus' == 'Success' means speech was found
            status = stt_data.get("RecognitionStatus", "")
            if status == "Success":
                # Prefer the NBest DisplayText (highest confidence)
                nbest = stt_data.get("NBest", [])
                if nbest:
                    full_transcript = nbest[0].get("Display", stt_data.get("DisplayText", ""))
                else:
                    full_transcript = stt_data.get("DisplayText", "")
            elif status in ("NoMatch", "InitialSilenceTimeout", "BabbleTimeout"):
                full_transcript = "Audio received, but no clear spoken words were recognized."
            else:
                full_transcript = f"Speech recognition returned status: {status}"
        else:
            raise RuntimeError(
                f"Azure Speech REST API error {stt_response.status_code}: {stt_response.text[:300]}"
            )

        if not full_transcript:
            full_transcript = "Audio received, but no clear spoken words were recognized."

        # Generate an AI ticket summary from the transcript
        ticket_summary = ""
        if full_transcript and "no clear spoken words" not in full_transcript:
            ai_summary_prompt = f"""Analyze this transcribed customer support call/voicemail and generate:
1. Customer Problem Summary (2-3 sentences)
2. Customer Sentiment during the call
3. Key Action Items & Next Steps for the Support Team

TRANSCRIPT:
{full_transcript}"""
            try:
                ticket_summary = text_response(ai_summary_prompt, system="You are a Support Quality Assurance & Triage AI.")
            except Exception:
                ticket_summary = "AI summary generation skipped."

        return jsonify(transcript=full_transcript, summary=ticket_summary)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# 5. User Manual & Invoice / Warranty Document Understanding
@app.get("/manual-analysis")
def manual_analysis():
    return render_template("manual_analysis.html")


@app.post("/api/manual-analysis")
def api_manual_analysis():
    try:
        file = request.files.get("file")
        analyzer = request.form.get("analyzer", "prebuilt-documentSearch")

        if not file:
            return jsonify(error="Please upload a document, manual, receipt, or warranty PDF/image."), 400

        if not CONTENT_ENDPOINT or not CONTENT_KEY:
            return jsonify(error="Azure Content Understanding endpoint or key is not configured in .env."), 400

        url = f"{CONTENT_ENDPOINT}/contentunderstanding/analyzers/{analyzer}:analyze?api-version={CONTENT_API_VERSION}"
        headers = {"Ocp-Apim-Subscription-Key": CONTENT_KEY}
        files = {"file": (file.filename, file.stream, file.mimetype or "application/pdf")}

        res = requests.post(url, headers=headers, files=files, timeout=180)
        if not res.ok:
            return jsonify(error=f"Content Understanding service returned status {res.status_code}: {res.text}"), res.status_code

        return jsonify(result=res.json())
    except Exception as e:
        return jsonify(error=str(e)), 500


# 6. Automated Escalation & Diagnostic Agent
@app.get("/support-agent")
def support_agent():
    return render_template("support_agent.html")


@app.post("/api/support-agent")
def api_support_agent():
    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        product_model = data.get("product_model", "").strip()
        serial_number = data.get("serial_number", "").strip()
        purchase_date = data.get("purchase_date", "").strip()

        if not message:
            return jsonify(error="Please provide issue symptoms or diagnostic response."), 400

        ticket_id = f"T2-ENG-{uuid.uuid4().hex[:6].upper()}"

        system_prompt = (
            "You are the 'Senior Diagnostic & Tier-2 Escalation Specialist'. "
            "Your job is to guide customers through deep technical diagnostics, evaluate warranty eligibility, "
            "isolate root causes, and generate an official, structured Tier-2 Engineering Escalation Ticket."
        )

        agent_prompt = f"""PRODUCT CONTEXT:
- Model: {product_model or 'Not provided'}
- Serial Number: {serial_number or 'Not provided'}
- Purchase Date: {purchase_date or 'Not provided'}

CUSTOMER INPUT / DIAGNOSTIC RESPONSE:
{message}

INSTRUCTIONS:
1. Provide a step-by-step diagnostic evaluation.
2. If the problem is unresolved, create a formal [TIER-2 ESCALATION TICKET] with:
   - Ticket ID: {ticket_id}
   - Severity & Priority
   - Warranty Status Evaluation (based on purchase date or typical 1-year standard warranty)
   - Steps Already Attempted
   - Specific Engineering Action Required
3. Keep the output clean, structured with markdown tables and bullet points."""

        reply = text_response(agent_prompt, system=system_prompt)

        # Persist escalation to SQLite DB
        try:
            priority = "Critical" if ("Critical" in reply or "Hazard" in reply) else "High"
            summary_snippet = f"{product_model or 'Hardware'}: {message[:100]}..."
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                INSERT INTO tickets (ticket_id, product_model, category, priority, sentiment, summary, details, source, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticket_id, product_model, "Tier-2 Escalation", priority, "Escalated", summary_snippet, reply, "Escalation Agent", "Escalated"))
            conn.commit()
            conn.close()
        except Exception:
            pass

        return jsonify(reply=reply, ticket_id=ticket_id)
    except Exception as e:
        return jsonify(error=str(e)), 500


# 7. Ticket Management & History Dashboard
@app.get("/tickets")
def tickets_page():
    return render_template("tickets.html")


@app.get("/api/tickets")
def get_tickets():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM tickets ORDER BY created_at DESC")
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(tickets=rows)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/tickets/<ticket_id>/status")
def update_ticket_status(ticket_id):
    try:
        data = request.json or {}
        new_status = data.get("status", "Open")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE tickets SET status = ? WHERE ticket_id = ?", (new_status, ticket_id))
        conn.commit()
        conn.close()
        return jsonify(success=True, ticket_id=ticket_id, status=new_status)
    except Exception as e:
        return jsonify(error=str(e)), 500


# Health Check & Configuration Overview
@app.get("/health")
def health():
    return jsonify(
        app_name="Product Support Assistant AI",
        status="operational",
        azure_openai={
            "configured": bool(AOAI_ENDPOINT and AOAI_KEY),
            "text_model": TEXT_MODEL,
            "vision_model": VISION_MODEL
        },
        speech_service={
            "configured": bool(SPEECH_KEY and SPEECH_ENDPOINT),
            "region": SPEECH_REGION
        },
        content_understanding={
            "configured": bool(CONTENT_KEY and CONTENT_ENDPOINT),
            "api_version": CONTENT_API_VERSION
        }
    )


@app.get("/uploads/<filename>")
def get_uploaded_file(filename):
    file_path = os.path.join(os.path.dirname(__file__), "uploads", filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify(error="File not found"), 404


if __name__ == "__main__":
    print("🚀 Starting Product Support Assistant AI Server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host="127.0.0.1", port=5000)
