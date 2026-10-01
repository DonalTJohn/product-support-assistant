# Product Support Assistant AI — Azure AI & Flask Studio

An enterprise-ready **Product Support Assistant** web application powered by **Azure OpenAI**, **Azure AI Foundry**, **Azure Speech Services**, and **Azure Content Understanding**.

---

## 🌟 Modules & Features

| Module | Route | Azure Service & Capability |
| :--- | :--- | :--- |
| **Customer Support Chat** | `/chat` | Empathetic, multi-turn conversational support assistant for FAQs & warranty guidance |
| **Ticket & Review Triage** | `/ticket-analysis` | NLP extraction: Issue Category, Urgency SLA, Sentiment/Churn Risk, Draft Reply |
| **Visual Diagnostics** | `/visual-troubleshoot` | Multimodal Vision inspection of damaged hardware, broken connectors & error screens |
| **Voice Call Transcriber** | `/voice-support` | Azure Speech REST transcription + AI-generated ticket summaries and action items |
| **Manuals & Invoice Reader** | `/manual-analysis` | Azure Content Understanding document analyzer for warranty & spec ingestion |
| **Tier-2 Escalation Agent** | `/support-agent` | Diagnostic isolation, warranty calculation, and automated Tier-2 Escalation Tickets |
| **Health Monitor** | `/health` | Live verification of Azure API keys, endpoints, and active model deployments |

---

## 🚀 Quick Setup & Run

### 1. Prerequisites
- Python 3.10+
- Azure OpenAI resource with a deployed model (e.g. `gpt-4.1-mini` or `gpt-4o`)
- (Optional) Azure Speech Service & Azure Content Understanding endpoints

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your Azure endpoints and API keys:
```bash
cp .env.example .env
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**.

---

## 📁 Directory Structure

```
ProductSupport/
├── .env.example              # Environment variable template
├── .env                      # Active Azure keys & endpoints
├── app.py                    # Flask server & Azure AI integration routes
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── uploads/                  # Temporary uploaded media files
└── templates/
    ├── base.html             # Main layout, typography & navigation
    ├── home.html             # Overview dashboard
    ├── chat.html             # Support chatbot UI
    ├── ticket_analysis.html  # Ticket triage & NLP analysis
    ├── visual_troubleshoot.html # Multimodal visual inspection
    ├── voice_support.html    # Speech-to-text call summarizer
    ├── manual_analysis.html  # Content Understanding reader
    └── support_agent.html    # Tier-2 Diagnostic & Escalation agent
```
