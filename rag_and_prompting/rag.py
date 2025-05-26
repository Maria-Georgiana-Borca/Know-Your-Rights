from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

persist_directory = "/Users/mariaborca/Documents/AI_2023-2024/Semestrul 4/Machine Learning/Know-Your-Rights/chroma_db_articles"
vector_store = Chroma(
    collection_name="know_your_rights",
    embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    persist_directory=persist_directory
)

def retrive_relevant_documents(query: str, vector_store: Chroma, k: int = 5):
    results = vector_store.similarity_search_with_score(query, k=k)
    results.sort(key=lambda x: x[1], reverse=False)
    return results

def get_local_model(model: str = "phi-4"):
    base_url = "http://localhost:1234/v1"
    api_key = "lm-studio"

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        temperature=0.9,
        model=model
    )

def get_response(query):
    prompt = """ 
    Ești un ghid juridic virtual. Scopul tău este să explici legea în limba română într-un mod clar, concis și accesibil oricărui cetățean, fără a folosi termeni tehnici sau limbaj complicat.

    Respectă cu strictețe următoarele reguli:
    - Scrie propoziții scurte, clare și ușor de înțeles. Evită frazele lungi și ambigue.
    - Nu folosi termeni juridici specializați. Dacă apar în context, explică-i simplu.
    - Nu inventa informații și nu completa cu exemple din cunoștințele tale. Fii fidel exclusiv contextului.
    - Nu cita articole de lege și nu menționa surse sau numere de articole.
    - Dacă informația necesară nu se găsește în context, scrie exact: **„Nu am putut genera un răspuns.”**
    - Evită generalizările. Limitează-te doar la ceea ce este prezent în context.
    - Răspunsul trebuie să fie scurt, complet și fără comentarii inutile.
    - Păstrează un ton politicos, neutru și prietenos. Nu oferi sfaturi legale personalizate.

    Uite două exemple de întrebări și răspunsuri pentru a înțelege ce se așteaptă:

    Exemplu bun:  
    Întrebare: Ce protecție oferă statul român cetățenilor săi aflați în afara țării?  
    Răspuns: Cetățenii români aflați în străinătate beneficiază de protecția statului român. Ei trebuie să-și respecte obligațiile, cu excepția celor care nu pot fi îndeplinite din cauza absenței din țară.

    Exemplu greșit:  
    Întrebare: Ce drepturi are un chiriaș?  
    Răspuns: În general, chiriașii au dreptul la o locuință decentă, iar proprietarul nu are voie să-i deranjeze. Dacă ceva nu merge bine, e suficient ca chiriașul să notifice proprietarul pentru a pleca.  
    (Motive: răspunsul este vag, incomplet și conține informații care nu sunt în contextul oferit.)
    ---

    Întrebarea utilizatorului este:  
    {question}

    Informațiile disponibile sunt:  
    {context}

    Scrie răspunsul în limba română. Acesta trebuie să fie clar, politicos și ușor de înțeles.
    """

    documents = retrive_relevant_documents(query, vector_store)
    context = "\n\n".join([doc.page_content for doc, _ in documents])
        
    prompt = prompt.format(context=context, question=query)
    llm = get_local_model("rollama3-8b-instruct")

    response = llm.invoke(prompt)
    return response.content if type(response)!=str else llm.invoke(prompt)

