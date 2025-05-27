import os
import logging
from pathlib import Path
from typing import List, Tuple, Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import DisplayConfig

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIR = "/Users/mariaborca/Documents/AI_2023-2024/Semestrul 4/Machine Learning/Know-Your-Rights/chroma_db_articles"
PROMPT_TEMPLATE = """ 
    Ești un ghid juridic virtual. Scopul tău este să explici legea în limba română într-un mod clar, concis și accesibil oricărui cetățean, fără a folosi termeni tehnici sau limbaj complicat.

    Respectă cu strictețe următoarele reguli:
    - Scrie propoziții scurte, clare și ușor de înțeles. Evită frazele lungi și ambigue.
    - Dacă în context apar termeni juridici, incearca sa ii explici pe intelesul tuturor, cu exemple dacă e necesar.
    - **Folosește exclusiv informațiile din context. Nu adăuga detalii din cunoștințele tale generale, experiență sau logică personală.**
    - **Fiecare afirmație trebuie să fie sprijinită clar de context. Dacă nu este, nu o include.**
    - Dacă informația necesară nu se găsește în context, scrie exact: **„Nu am putut genera un răspuns.”**
    - Nu cita articole de lege, nu menționa surse și nu include numere de articole.
    - Răspunsul trebuie să fie scurt, complet și fără comentarii inutile.
    - Păstrează un ton politicos, neutru și prietenos. Nu oferi sfaturi legale personalizate.

    IMPORTANT:
    - **Nu încerca să completezi lipsurile. Dacă informația lipsește, spune asta.**
    - **Nu generaliza și nu introduce interpretări.**

    ---

    Uite două exemple de întrebări și răspunsuri pentru a înțelege ce se așteaptă:

    *Exemplu bun:*  
    Întrebare: Ce protecție oferă statul român cetățenilor săi aflați în afara țării?  
    Răspuns: Cetățenii români aflați în străinătate beneficiază de protecția statului român. Ei trebuie să-și respecte obligațiile, cu excepția celor care nu pot fi îndeplinite din cauza absenței din țară.  
    Pe scurt: statul îi protejează, dar au și obligații.

    *Exemplu greșit:*  
    Întrebare: Ce drepturi are un chiriaș?  
    Răspuns: În general, chiriașii au dreptul la o locuință decentă, iar proprietarul nu are voie să-i deranjeze. Dacă ceva nu merge bine, e suficient ca chiriașul să notifice proprietarul pentru a pleca.  
    (Motive: răspunsul este vag, incomplet și conține informații care nu sunt în contextul oferit.)

    ---

    Întrebarea utilizatorului este:  
    {question}

    Informațiile disponibile sunt:  
    {context}

    Scrie răspunsul în limba română. Acesta trebuie să fie clar, politicos și ușor de înțeles. Rămâi strict la informațiile din context. Dacă este relevant, adaugă o propoziție de concluzie simplificată.
"""

# Get Vector Store
vector_store = Chroma(
    collection_name="know_your_rights",
    embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    persist_directory=PERSIST_DIR
)

# Display configuration for evaluation
display_config = DisplayConfig(
    print_results=False,
    show_indicator=False
)

def retrive_relevant_documents(query: str, vector_store: Chroma, k: int = 3) -> List[Tuple]:
    """Retrieves relevant documents from the vector store based on the query."""
    docs = vector_store.similarity_search_with_score(query, k=k)
    return sorted(docs, key=lambda x: x[1])

def evaluate_faithfulness(
    query: str,
    answer: str,
    context: str,
) -> float:
    """Evaluates the faithfulness of the model's response."""
    metric = FaithfulnessMetric(threshold=0.7, verbose_mode=False, model="gpt-4o-mini")
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context
    )
    results = evaluate(
        test_cases=[test_case],
        metrics=[metric],
        display_config=display_config
    )

    for tr in results.test_results:
        for metric in tr.metrics_data:
            if metric.name == "Faithfulness":
                logger.info(f"Faithfulness score: {metric.score} (pass: {metric.success})")
                logger.info(f"Reason: {metric.reason}")
                return metric.score
    return 0.0

def get_local_llm(model: str = "rollama3-8b-instruct") -> ChatOpenAI:
    """Returns a local LLM instance."""
    return ChatOpenAI(
        base_url=LOCAL_LLM_BASE_URL,
        api_key=LOCAL_LLM_API_KEY,
        temperature=0,
        model=model
    )

def generate_response(
    query: str,
    llm: ChatOpenAI,
    k: int = 3
) -> str:
    """Generate a single response using context retrieval."""
    docs = retrive_relevant_documents(query, vector_store, k)
    context = "\n\n".join(doc.page_content for doc, _ in docs)
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)
    response = llm.invoke(prompt)
    return getattr(response, "content", response)

def generate_with_retries(
    query: str,
    llm: ChatOpenAI,
    base_k: int = 3,
    max_retries: int = 3,
    threshold: float = 0.7
) -> str:
    """Attempt generation multiple times and return the best-faithful answer."""
    best_score = 0.0
    best_answer = ""

    for attempt in range(max_retries):
        k = base_k + (2 if attempt > 0 else 0)
        logger.info("Attempt %d with k=%d", attempt + 1, k)

        answer = generate_response(query, llm, k)
        score = evaluate_faithfulness(
            query,
            answer,
            [doc.page_content for doc, _ in retrive_relevant_documents(query, vector_store, k)]
        )

        # logger.info("Faithfulness score: %.2f", score)
        if score >= threshold:
            logger.info("Threshold met (%.2f) on attempt %d; returning answer", score, attempt + 1)
            return answer

        if score > best_score:
            best_score = score
            best_answer = answer

    logger.warning(
        "Threshold not met (%.2f) after %d tries; returning best answer (score %.2f)",
        threshold,
        max_retries,
        best_score
    )
    return best_answer