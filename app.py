import os
import base64
import uuid
import requests
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB max upload

# Ensure uploads folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), "uploads"), exist_ok=True)

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
        return jsonify(result=analysis_result)
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
    try:
        file = request.files.get("audio")
        if not file:
            return jsonify(error="Please upload a valid audio recording (.wav, .mp3, .m4a, .ogg)."), 400

        if not SPEECH_KEY or not SPEECH_ENDPOINT:
            return jsonify(error="Azure Speech Service endpoint or key is not configured in .env."), 400

        language = request.form.get("language", "en-US")
        url = f"{SPEECH_ENDPOINT}/speechtotext/v3.2/transcriptions:transcribe?api-version=2024-11-15"
        headers = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}
        files = {"audio": (file.filename, file.stream, file.mimetype or "audio/wav")}
        data = {"definition": f'{{"locales":["{language}"],"profanityFilterMode":"Masked"}}'}

        speech_req = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        
        if not speech_req.ok:
            return jsonify(error=f"Azure Speech Service returned status {speech_req.status_code}: {speech_req.text}"), speech_req.status_code

        speech_data = speech_req.json()
        
        # Extract transcribed text phrases
        phrases = []
        combined_phrases = speech_data.get("combinedPhrases", [])
        if combined_phrases:
            for item in combined_phrases:
                if "text" in item:
                    phrases.append(item["text"])
        
        full_transcript = " ".join(phrases) if phrases else "Audio transcribed successfully (see raw payload)."

        # If transcript extracted, generate an automated ticket summary using AI
        ticket_summary = ""
        if full_transcript and full_transcript != "Audio transcribed successfully (see raw payload).":
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

        return jsonify(
            transcript=full_transcript,
            summary=ticket_summary,
            raw_response=speech_data
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


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

        system_prompt = (
            "You are the 'Senior Diagnostic & Tier-2 Escalation Specialist'. "
            "Your job is to guide customers through deep technical diagnostics, evaluate warranty eligibility, "
            "isolate root causes, and when necessary, generate an official, structured Tier-2 Engineering Escalation Ticket."
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
   - Ticket ID (auto-generate e.g. T2-SUP-XXXXX)
   - Severity & Priority
   - Warranty Status Evaluation (based on purchase date or typical 1-year standard warranty)
   - Steps Already Attempted
   - Specific Engineering Action Required
3. Keep the output clean, structured, and easy for both customer and Tier-2 engineers to read."""

        reply = text_response(agent_prompt, system=system_prompt)
        return jsonify(reply=reply)
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
