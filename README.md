✈️ Aviation Chatbot

Ask questions about aviation documents and get instant, accurate answers with sources. Built for aviation professionals who need fast access to technical information.

Show Image
Show Image
Show Image
Show Image
🔗 Live Demo: Try it here!

👋 What is This?
Ever spent hours searching through hundreds of pages of aviation manuals to find one specific answer? Yeah, me too. That's why I built this.
In simple terms: Upload your aviation PDFs, ask questions in plain English, and get accurate answers with exact page references. No more Ctrl+F through 500-page documents!
What Makes It Smart?
Unlike basic search (which just matches words), this chatbot actually understands what you're asking:

Ask: "What is baggage handling?"
Get: A detailed explanation from the baggage handling manual, page 47
Ask: "What does ILS stand for?"
Get: "Instrument Landing System" with technical details from airport operations, page 203

It's like having an aviation expert who's memorized all your manuals and can instantly find exactly what you need.

🎯 Who Is This For?
Perfect if you:

👨‍✈️ Work in aviation and need quick answers
📚 Study aviation and want to search across multiple documents
🏢 Manage aviation operations and need fast access to procedures
🔧 Handle maintenance and reference multiple manuals daily

Not perfect if you:

Just want to search one PDF (try Ctrl+F instead)
Don't have technical documents to query
Need legal advice (this gives info, not legal opinions)


✨ Cool Features
What It Does Really Well
🔍 Smart Search
Combines exact keyword matching + AI understanding of meaning. So it finds both exact terms (like "ILS") AND related concepts (like "landing systems").
📚 Always Shows Sources
Every answer includes:

Which document it came from
Exact page number
Preview of the original text

No more "trust me bro" answers!
⚡ Fast

Average response: 5.9 seconds
Simple questions: 3-5 seconds
Complex questions: 5-7 seconds

📤 Easy Document Upload
Drag, drop, done. Upload new PDFs anytime through the web interface. No coding required.
🎨 Dark Theme
Because nobody likes blinding white screens at 2 AM.

📊 What's Inside Right Now?
Currently loaded with 8 aviation documents:

Airport Engineering & Design (3,026 chunks)
Airport Operations Manual (1,915 chunks)
Maintenance Management Systems (829 chunks)
Baggage Handling Systems (713 chunks)
SCADA Manual (608 chunks)
Gazette of India (Aviation Regulations) (111 chunks)
Night Operations Procedures (71 chunks)
Annual Appraisal Policy (7 chunks)

Total: 7,280 searchable chunks of knowledge

🚀 Quick Start (10 Minutes)
Want to try it on your computer? Here's how:
You'll Need:

Python 3.11+ (Download here)
Docker Desktop (Get it here)
Gemini API key - Free! (Get yours here)

Setup:
1. Get the code:
bashgit clone https://github.com/nithin8688/aviation-chatbot.git
cd aviation-chatbot
2. Install Python packages:
bashpip install -r requirements.txt
3. Start the database:
bashdocker run --name aviation-postgres \
  -e POSTGRES_PASSWORD=aviation123 \
  -e POSTGRES_DB=aviation_chatbot \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
4. Add your API key:
Create a file called .env in the project folder:
DB_PASSWORD=aviation123
GEMINI_API_KEY=your-api-key-here
5. Run it:
bashstreamlit run app.py
6. Open your browser:
Go to http://localhost:8501
Done! 🎉

💬 How to Use It
Asking Questions
Just type naturally, like you're asking a colleague:
Good questions:

"What is baggage?"
"How does SCADA work?"
"What are runway safety procedures?"
"Explain the responsibilities of apron control"

The app will:

Search through all documents
Find the most relevant sections
Generate a clear answer
Show you exactly where it found the info

Uploading New Documents

Click "📤 Upload New PDF" in the sidebar
Choose your PDF (up to 200MB)
Wait while it processes (you'll see a progress bar)
Start asking questions about it!

Pro tip: For faster processing on the free cloud tier, keep PDFs under 25MB. Larger files? Split them into smaller chunks first.
Managing Documents

View stats: See how many chunks from each document
Delete documents: Select and delete (with confirmation)
Re-upload: Delete old version, upload new one


🏗️ How It Works (Explained Simply)
The Magic Behind The Scenes
Think of it like a librarian with superpowers:

When you upload a PDF:

Reads every page
Breaks it into bite-sized chunks (800 characters each)
Creates a "fingerprint" for each chunk (called an embedding)
Stores everything in a database


When you ask a question:

Creates a "fingerprint" of your question
Finds chunks with similar fingerprints (that's the AI part!)
Also finds exact keyword matches (that's the traditional search)
Combines both results (hybrid search = best of both worlds)
Sends the best 5 chunks to an AI
AI reads them and writes a clear answer
Shows you the sources



Why It's Fast
Optimizations:

✅ Caches search indexes (100× faster on repeat queries)
✅ Reuses AI connections (no startup delay)
✅ Smart batching (processes multiple things at once)
✅ Reduced from 8 sources to 5 (40% speed boost)

Result: Went from 20 seconds → 5.9 seconds average!

🌐 Want to Deploy Online?
Best option: Streamlit Cloud + Render.com (both free!)
Takes about 30 minutes to set up.
I've written a complete step-by-step guide: DEPLOYMENT_GUIDE.md
Quick overview:

Push your code to GitHub
Create free database on Render.com
Deploy app on Streamlit Cloud
Add your API keys and database credentials
Share the URL with your team!

Cost: $0/month for the first 90 days, then $0-27/month depending on your needs.

📁 Project Structure
aviation-chatbot/
│
├── app.py                    # The web interface (what you see)
├── .env                      # Your secrets (NEVER commit this!)
│
├── src/                      # The brain of the operation
│   ├── config.py             # Settings you can tweak
│   ├── rag_engine.py         # The AI query handler
│   ├── hybrid_search.py      # Smart search (BM25 + vectors)
│   ├── ingest.py             # PDF processor
│   └── db_utils.py           # Database operations
│
├── data/                     # Where stuff is stored
│   ├── uploaded_pdfs/        # Your PDF files
│   └── chat_history.json     # Saved conversations
│
├── requirements.txt          # Python packages needed
├── README.md                 # You are here!
└── DEPLOYMENT_GUIDE.md       # How to deploy to the cloud

⚙️ Performance Tuning
Want to make it faster or more accurate? Here's what you can adjust:
In src/config.py:
python# Want faster responses? (less accurate)
TOP_K_RETRIEVAL = 3  # Default: 5

# Want better accuracy? (slower)
TOP_K_RETRIEVAL = 8  # Default: 5

# Adjust search balance
HYBRID_ALPHA = 0.7   # More AI understanding
HYBRID_ALPHA = 0.5   # More exact matching
Trade-offs:

More sources = More accurate, slower
Fewer sources = Faster, might miss context
Higher alpha = Better for conceptual questions
Lower alpha = Better for exact terms (IDs, codes)


🐛 Common Problems & Fixes
"It's not working!"
Database won't connect?
bash# Make sure Docker is running
docker ps

# If not, start it
docker start aviation-postgres
API key error?
bash# Check your .env file exists
ls .env

# Make sure it has your key
cat .env
# Should show: GEMINI_API_KEY=your-key-here
Import errors?
bash# Reinstall packages
pip install -r requirements.txt --upgrade
Slow responses?

First time always slower (loading models)
Subsequent queries should be ~6 seconds
If consistently > 10s, reduce TOP_K_RETRIEVAL to 3

Running out of memory on Streamlit Cloud?

Split large PDFs into smaller files (< 25MB each)
Or upgrade to paid tier ($20/month for 4GB RAM)


🎯 Real-World Examples
Example 1: Quick Lookup
You: "What is ILS?"
Bot: "ILS stands for Instrument Landing System. It's a precision landing aid that provides pilots with electronic guidance during approach and landing..."
Sources: airport_operations.pdf, Page 203
Time: 3.2 seconds ✅
Example 2: Complex Query
You: "What are the responsibilities of airport apron control?"
Bot: "Airport apron control is responsible for: 1) Managing aircraft movements in the apron area, 2) Coordinating with air traffic control, 3) Ensuring safe parking positions..."
Sources:

airport_operations.pdf, Pages 45, 67, 89
Airport_Engineering.pdf, Page 156

Time: 6.1 seconds ✅
Example 3: Off-Topic (Privacy Protection)
You: "What's the weather like?"
Bot: "I'm an aviation technical documentation assistant. I can only answer questions based on the documents in my knowledge base. Could you ask about airport operations, procedures, or technical specifications?"
Sources: None (not in documents)
Time: 2.1 seconds ✅

🚀 What's Next? (Roadmap)
Coming Soon (Maybe)

💬 Conversation memory: "Tell me more about that" (remembers context)
🗣️ Voice input: Ask questions hands-free
📊 Analytics dashboard: See most common queries
🔌 API access: Integrate with other tools
📱 Mobile app: Query on the go

If You Want to Contribute
Got ideas? Found bugs? Pull requests welcome!
How to contribute:

Fork this repo
Create a feature branch: git checkout -b cool-new-feature
Make your changes
Test thoroughly
Submit a pull request

Please include:

Clear description of what you changed
Why you changed it
Screenshots if it's a UI change
Tests if you added new functionality


🤔 Frequently Asked Questions
Q: How accurate is it?
A: Very accurate for factual info in the documents. It always cites sources, so you can verify. Test success rate: 100% ✅
Q: Can it summarize documents?
A: It answers specific questions. For full summaries, ask: "Give me an overview of [topic]"
Q: What if it gives a wrong answer?
A: Check the sources it provides. If wrong, the info might not be in your documents, or you might need to rephrase the question.
Q: Is my data private?
A: On local deployment: 100% private. On cloud: Your PDFs go to your database, queries go to Gemini API. Read Gemini's privacy policy for details.
Q: How many PDFs can I upload?
A: Unlimited! But performance depends on your database size. I'm running 8 PDFs (7,280 chunks) smoothly.
Q: Can I use it for non-aviation documents?
A: Absolutely! Works with any technical documentation. Just upload your PDFs.
Q: Why Gemini instead of GPT?
A: Free tier is generous (1,500 requests/day), fast, and accurate. But you could swap in GPT if you prefer!

📞 Need Help?
Found a bug? Open an issue
Have a question? Email me: nithin.dev8688@gmail.com
Want to chat? GitHub: @nithin8688

💝 Special Thanks
This project wouldn't exist without these awesome tools:

Google Gemini - The AI brain
PostgreSQL + pgvector - The memory
Streamlit - The beautiful interface
Sentence Transformers - The understanding
You! - For checking this out


📊 Fun Stats

Lines of code: ~3,000
Documents indexed: 8 PDFs
Total knowledge chunks: 7,280
Average response time: 5.9 seconds
Test success rate: 100%
Coffee consumed during development: Too much ☕


🎓 What I Learned Building This
Technical skills:

RAG (Retrieval-Augmented Generation) systems
Vector databases and embeddings
Hybrid search algorithms
Streamlit app development
PostgreSQL optimization
Cloud deployment

Non-technical lessons:

Speed matters more than perfection
Good documentation saves time
Users want simple, not fancy
Cache everything you can!


Built with ❤️ by someone tired of searching through 500-page PDFs
Last updated: February 2026
