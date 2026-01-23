PROJECT DROPKICK
================

Project Dropkick is an advanced conversational AI prototype designed to simulate emotionally resonant interactions with a fully characterized persona. This project explores the concept of AI companions that feel believable, emotionally aware, and capable of forming long-term relationships with users.

For the purposes of this project, Eric King — a character from "The Dark Pictures Anthology: House of Ashes" (Supermassive Games, 2021) — has been reimagined as a persistent, evolving AI companion. The system brings this fictional character into an interactive form, transforming him into a human-like digital presence.

The architecture is built for flexibility: developers, writers, and researchers can easily substitute Eric with any custom persona through editable configuration files and prompt structures.

Note: This is an early ALPHA version of the project. Many features are still in development.

------------------------------------------------------------
NOTABLE FEATURES
------------------------------------------------------------

- Conversational AI Agent:
  Uses OpenAI's GPT-4 or GPT-3.5 to simulate natural, human-like conversations.

- Dual Memory System (Short-Term + Long-Term):
  Remembers recent context during conversation (short-term) while storing all past interactions in a database (long-term) for persistent memory.

- Identity-Driven Behavior:
  The AI adheres to a defined persona through three sources:
    - character_info.json (traits, backstory, personality)
    - app.py (custom prompt structure)
    - dialogues.txt (real or custom sample dialogue)

- Character Simulation:
  This version features Eric King, whose background, tone, and personality are hard-coded into the AI to maintain authenticity.

- Easy Customization:
  Swap Eric for any character by editing config files and text samples (explained later).

------------------------------------------------------------
INSTALLATION
------------------------------------------------------------

1. Clone the repository:

   git clone https://github.com/shay0j/Project-Dropkick
   
   cd Project-Dropkick

2. (Optional) Create and activate a virtual environment:

   python -m venv venv

   - Windows:
     venv\Scripts\activate

   - macOS/Linux:
     source venv/bin/activate

3. Install required libraries:

   pip install -r requirements.txt

4. Add your OpenAI API key:
   - Rename the file ".env.example" to ".env"
   - Open ".env" and replace the placeholder with your key:

     OPENAI_API_KEY=your_openai_api_key_here

5. Run the app:

   python app.py

6. Open your browser and go to:

   http://127.0.0.1:5000/

   You’ll see a chat window where you can start talking to Eric.

------------------------------------------------------------
USAGE
------------------------------------------------------------

This AI is designed to simulate a long-term, emotionally intelligent virtual character. It can be used as:

- A Companion AI:
  Interacts naturally, remembers your past, and evolves over time.

- A Helpdesk Chatbot:
  Acts as a support assistant with a consistent personality and memory of user issues.

- A Character in Games or Roleplay:
  Creates interactive, persistent personalities for storytelling or immersion.

You can easily replace the default character with your own.

------------------------------------------------------------
MEMORY SYSTEM
------------------------------------------------------------

The AI uses both short-term and long-term memory:

1. SHORT-TERM MEMORY:
   - Fetches the latest 150 messages from memory.
   - These are sent to OpenAI to maintain coherent conversation.
   - Handled in app.py (send_message function).

2. LONG-TERM MEMORY:
   - All messages are stored in memory.db (SQLite).
   - Enables memory across app restarts and long-term user relationships.
   - Handled in memory_db.py.

   Database Schema:
     memory (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         user_id TEXT,
         role TEXT,
         content TEXT
     )

   Functions:
     - add_memory(user_id, role, content)
     - get_memory(user_id, limit=50)

------------------------------------------------------------
CUSTOMIZATION
------------------------------------------------------------

To replace Eric King with your own character:

1. Open character_info.json:
   - Replace the name, bio, traits, etc. with your new character.

2. Edit dialogues.txt:
   - Write natural sample lines for your character’s voice and tone.

3. Modify the system prompt in app.py:
   - Update the section that defines behavior and rules.
   - Adjust how memory is included, how the AI should behave, and what tone to follow.

4. (Optional) Change the model:
   - Open config.ini and set your preferred model:

     model = gpt-4     (or gpt-3.5-turbo)

------------------------------------------------------------
DISCLAIMER & CREDITS
------------------------------------------------------------

This project features the character Eric King from "The Dark Pictures Anthology: House of Ashes" (2021), developed by Supermassive Games and published by Bandai Namco Entertainment.

This project is not affiliated with, endorsed by, or sponsored by Supermassive Games or Bandai Namco.

All rights to the characters, artwork, and story belong to their respective owners.

This project is intended strictly for personal, educational, and non-commercial use.

------------------------------------------------------------
AUTHOR
------------------------------------------------------------

Created and maintained by: Ola Borowiec

GitHub: github.com/shay0joestar  
Email: ola.borowiec@yahoo.com

Feel free to reach out with ideas, improvements, or collaborations.

------------------------------------------------------------
ACKNOWLEDGEMENTS
------------------------------------------------------------

Special thanks to Krzysztof Soczówka (GitHub: ksoczowka) for peer reviewing and providing valuable feedback on the project’s development.

------------------------------------------------------------
LICENSE
------------------------------------------------------------

YOU ARE ALLOWED TO:
- Use the software for personal, non-commercial purposes.
- Customize the character identity and behavior for your own use.

YOU ARE NOT ALLOWED TO:
- Sell, redistribute, or publicly publish this software.
- Use it for commercial purposes.
- Modify or reuse the code beyond character customization.

For all other use cases (including education, research, or commercial use), please contact the author directly.
