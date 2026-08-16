# ThermoLingo
Describe the physics. Watch it simulate.
ThermoLingo turns plain-English descriptions of heat-conduction problems into live physics simulations, using an LLM to parse specs and explain results grounded in real computed data.

Setup & running it locally

Requirements: Python 3.10+, a free Gemini API key.

bash
# 1. Clone the repo
git clone https://github.com/<your-username>/thermolingo.git
cd thermolingo

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# then open .env and paste in your own Gemini key

Get a free Gemini API key (no credit card required) at https://aistudio.google.com/apikey.

Your .env should look like:

OPENAI_API_KEY=your-gemini-api-key
OPENAI_MODEL=gemini-3.6-flash
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

Gemini model availability changes over time. If you get a 404 on the model name, run the model-listing snippet in the repo (list_models.py) with your key to see what's currently available, and update OPENAI_MODEL accordingly.

bash
# 5. Run it
python main.py

Open http://localhost:8000 in your browser. Click one of the example prompts or write your own description of a rod, then hit Run Simulation.
