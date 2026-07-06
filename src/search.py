import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL não encontrado no .env.")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embedding_client():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY não encontrado no .env.")

    return GoogleGenerativeAIEmbeddings(
        model=google_model,
        google_api_key=google_api_key,
    )


def get_chat_client():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_chat_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY não encontrado no .env.")

    return ChatGoogleGenerativeAI(
        model=google_chat_model,
        temperature=0.0,
        top_p=0.95,
    )


def vector_to_pg_literal(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def get_top_k_context(question, top_k=10):
    embedding_client = get_embedding_client()
    query_embedding = embedding_client.embed_documents([question])

    if not query_embedding or len(query_embedding) == 0:
        raise RuntimeError("Falha ao gerar embedding para a pergunta.")

    query_vector = query_embedding[0]
    query_vector_literal = vector_to_pg_literal(query_vector)

    engine = create_engine(DATABASE_URL)
    sql = text(
        """
        SELECT id, source, chunk_index, content
        FROM public.documents
        ORDER BY embedding <-> CAST(:query_vector AS vector)
        LIMIT :k
        """
    )

    with engine.connect() as conn:
        result = conn.execute(sql, {"query_vector": query_vector_literal, "k": top_k})
        rows = result.fetchall()

    if not rows:
        return []

    context_items = []
    for row in rows:
        context_items.append(
            f"Fonte: {row.source} | Chunk: {row.chunk_index}\n{row.content.strip()}"
        )

    return context_items


def build_prompt(context_items, question):
    contexto = "\n\n---\n\n".join(context_items)
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)


def search_prompt():
    try:
        chat_client = get_chat_client()
        return {
            "chat_client": chat_client,
            "build_prompt": build_prompt,
            "get_top_k_context": get_top_k_context,
        }
    except Exception as exc:
        logger.error("Erro ao inicializar o chat de busca: %s", exc)
        return None


def answer_question(question, search_state):
    context_items = search_state["get_top_k_context"](question, top_k=10)

    if not context_items:
        return "Não tenho informações necessárias para responder sua pergunta."

    prompt = search_state["build_prompt"](context_items, question)
    messages = [
        SystemMessage(content="Você é um assistente que responde somente com base no contexto fornecido."),
        HumanMessage(content=prompt),
    ]

    response = search_state["chat_client"].invoke(messages)
    answer = getattr(response, "content", None) or str(response)
    answer = answer.strip()

    if not answer:
        return "Não tenho informações necessárias para responder sua pergunta."

    return answer
