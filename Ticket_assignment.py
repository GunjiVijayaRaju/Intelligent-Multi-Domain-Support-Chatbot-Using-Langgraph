from utils import *


genai.configure(api_key="AIzaSyBpB-NLQ50R0ftU9B-cNVEPzS0pY96OtMY")

# Initialize FAISS index
index = faiss.IndexFlatL2(768)  # 768 for Gemini Embedding

# --- DATABASE SETUP ---
def create_tables():
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        field TEXT,
        skills TEXT,
        completed_tasks TEXT,
        open_tickets INTEGER DEFAULT 0,
        availability TEXT
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT,
        field TEXT,
        issue_description TEXT,
        assigned_to INTEGER,
        status TEXT,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

# --- ADD SUPPORT PERSON ---
def add_person(name, email, field, skills, completed_tasks, availability="yes"):
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO persons (name, email, field, skills, completed_tasks, open_tickets, availability)
        VALUES (?, ?, ?, ?, ?, 0, ?)""",
        (name, email, field, skills, completed_tasks, availability))
    conn.commit()
    conn.close()

# --- ADD SUPPORT TICKET ---
def add_ticket(user_name, field, issue_description):
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO tickets (user_name, field, issue_description, assigned_to, status, created_at)
        VALUES (?, ?, ?, NULL, 'open', ?)""",
        (user_name, field, issue_description, created_at))
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ticket_id

# --- ENRICH TICKET VIA LLM ---
def enrich_ticket_with_tags(description):
    prompt = f"""
You are a support assistant. Normalize the user issue and extract support skill tags.
Return JSON in this format only return this strictly i am telling don't include any other extra matter or details i need onlt this details that's it:
{{
  "normalized_issue": "...",
  "tags": ["tag1", "tag2", ...]
}}

User Issue: {description}
    """
    model = llm
    res = model.invoke(prompt)
    content = res.content
    import json
    return json.loads(content)

# --- GET GEMINI EMBEDDING ---
def get_gemini_embedding(text):
    response = embedding.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    return response["embedding"]



def assign_ticket(ticket_id):
    conn = sqlite3.connect("support.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    ticket = cursor.fetchone()
    if not ticket:
        return "❌ Ticket not found."

    _, user_name, field, issue_description, _, _, _ = ticket
    enriched = enrich_ticket_with_tags(issue_description)
    issue_tags = enriched['tags']
    summary = enriched['normalized_issue']

    print(f"\n🎫 Ticket ID: {ticket_id}")
    print(f"   → Tags: {issue_tags}")
    print(f"   → Normalized: {summary}")

    # Get available persons
    cursor.execute("""
        SELECT id, name, email, skills, completed_tasks, open_tickets 
        FROM persons 
        WHERE field = ? AND availability = 'yes'
    """, (field,))
    available_persons = cursor.fetchall()

    if not available_persons:
        return "❌ No available agents."

    # Initialize a fresh FAISS index
    dim = 768  # Gemini embedding dimension
    index = faiss.IndexFlatL2(dim)
    person_map = {}

    for idx, person in enumerate(available_persons):
        id_, name, email, skills, completed_tasks, open_tickets = person
        combined_text = f"{skills}, {completed_tasks}"
        embedding_vec = get_gemini_embedding(combined_text)
        index.add(np.array([embedding_vec], dtype=np.float32))
        person_map[idx] = {
            "id": id_,
            "name": name,
            "email": email,
            "open_tickets": open_tickets
        }

    query_embedding = get_gemini_embedding(", ".join(issue_tags) or summary)
    k = min(5, len(person_map))
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k)

    best_candidate = None
    best_score = float("inf")

    for rank, idx in enumerate(I[0]):
        if idx == -1 or idx not in person_map:
            continue
        person = person_map[idx]
        adjusted_score = D[0][rank] + 0.5 * person['open_tickets']

        print(f"\n🔍 Candidate {rank + 1}:")
        print(f"   → Name: {person['name']}, Email: {person['email']}")
        print(f"   → Tickets: {person['open_tickets']}, Distance: {D[0][rank]:.2f}, Score: {adjusted_score:.2f}")

        if adjusted_score < best_score:
            best_score = adjusted_score
            best_candidate = person

    if best_candidate:
        cursor.execute("UPDATE tickets SET assigned_to = ? WHERE id = ?", (best_candidate['id'], ticket_id))
        cursor.execute("UPDATE persons SET open_tickets = open_tickets + 1 WHERE id = ?", (best_candidate['id'],))
        conn.commit()
        conn.close()
        return f"\n✅ Assigned To: {best_candidate['name']} and you can contact him through {best_candidate['email']}"
    else:
        conn.close()
        return "❌ No suitable agent found."

