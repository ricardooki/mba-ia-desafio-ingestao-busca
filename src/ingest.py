import os
import time
import logging
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

from sqlalchemy import Column, Integer, String, Text, create_engine, text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# Configura logging para monitorar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag"
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === MODELOS CORRETOS ===
# Use "models/embedding-001" para 1536 dimensões
# Use "models/text-embedding-004" para 3072 dimensões
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash")

# === CONFIGURAÇÕES MODIFICADAS PARA ECONOMIZAR TOKENS ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
MAX_BATCHES = int(os.getenv("MAX_BATCHES", "2"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "3"))
SLEEP_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "5.0"))
USE_CHAT_PROCESSING = os.getenv("USE_CHAT_PROCESSING", "false").lower() == "true"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", "2.0"))

# === DIMENSÃO DO EMBEDDING (será detectada automaticamente) ===
EMBEDDING_DIM = None

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # A dimensão será definida dinamicamente quando a classe for criada
    embedding = Column(Vector(1536), nullable=False)  # Valor padrão, será atualizado


def get_embedding_dimension():
    """Detecta a dimensão real do embedding fazendo um teste."""
    global EMBEDDING_DIM
    
    if EMBEDDING_DIM is not None:
        return EMBEDDING_DIM
    
    logger.info("🔍 Detectando dimensão do embedding...")
    
    try:
        # Cria um cliente temporário
        temp_client = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Gera um embedding de teste com um texto pequeno
        test_text = "Teste"
        test_result = temp_client.embed_documents([test_text])
        
        # Pega a dimensão real
        EMBEDDING_DIM = len(test_result[0])
        
        logger.info(f"✅ Modelo {GOOGLE_EMBEDDING_MODEL} gera vetores de {EMBEDDING_DIM} dimensões")
        
        # Atualiza a coluna 'embedding' da tabela com a dimensão correta
        Document.__table__.c.embedding.type = Vector(EMBEDDING_DIM)
        
        return EMBEDDING_DIM
        
    except Exception as e:
        logger.error(f"❌ Erro ao detectar dimensão: {e}")
        # Fallback para 1536 (padrão do embedding-001)
        EMBEDDING_DIM = 1536
        logger.warning(f"⚠️ Usando fallback: {EMBEDDING_DIM} dimensões")
        return EMBEDDING_DIM


def recreate_table_with_correct_dimension(engine):
    """Recria a tabela com a dimensão correta."""
    global EMBEDDING_DIM
    
    # Primeiro detecta a dimensão
    dim = get_embedding_dimension()
    
    try:
        with engine.connect() as conn:
            # Verifica se a tabela existe
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents'
                );
            """))
            table_exists = result.scalar()
            
            if table_exists:
                # Verifica a dimensão atual
                result = conn.execute(text("""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = 'documents'::regclass 
                    AND attname = 'embedding';
                """))
                current_dim = result.scalar()
                
                if current_dim:
                    current_dim = current_dim - 8  # Fórmula para vector
                    logger.info(f"Dimensão atual do banco: {current_dim}")
                    
                    if current_dim != dim:
                        logger.warning(f"⚠️ Dimensão incorreta! Atual: {current_dim}, Esperado: {dim}")
                        logger.warning("Recriando tabela com a dimensão correta...")
                        
                        # Drop e recreate
                        conn.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
                        conn.commit()
                        logger.info("✅ Tabela removida")
                        
                        # Atualiza a coluna 'embedding' da tabela e recria
                        Document.__table__.c.embedding.type = Vector(dim)
                        Base.metadata.create_all(engine)
                        logger.info(f"✅ Tabela recriada com Vector({dim})")
                        return True
                    else:
                        logger.info(f"✅ Dimensão correta: {dim}")
                        return True
            else:
                # Tabela não existe, cria com a dimensão correta
                logger.info("Criando tabela 'documents'...")
                # Atualiza a coluna antes de criar as tabelas
                Document.__table__.c.embedding.type = Vector(dim)
                Base.metadata.create_all(engine)
                logger.info(f"✅ Tabela criada com Vector({dim})")
                return True
                
    except Exception as e:
        logger.error(f"❌ Erro ao verificar/recriar tabela: {e}")
        return False


def get_embedding_client():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY não encontrado no .env.")
    return GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def get_chat_client():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY não encontrado no .env.")
    return ChatGoogleGenerativeAI(
        model=GOOGLE_CHAT_MODEL,
        temperature=0.3,
        top_p=0.95,
    )


def load_pdf_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"O arquivo PDF não foi encontrado: {path}")

    reader = PdfReader(path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"Página {page_number}\n{page_text}")

    return "\n\n".join(pages)


def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Divide o texto em chunks menores para economizar tokens."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def process_chunk_with_chat(chunk_text, chat_client):
    """Processa um chunk com chat (opcional - DESABILITADO POR PADRÃO)."""
    messages = [
        SystemMessage(content="Você é um assistente que prepara texto para embeddings."),
        HumanMessage(content=f"""
        Por favor, limpe e normalize o seguinte texto para prepará-lo para geração de embeddings.
        Mantenha o significado original, mas remova ruídos, formatação excessiva e normalize o texto.
        Mantenha informações importantes e estrutura lógica.
        
        Texto original:
        {chunk_text}
        
        Texto limpo:
        """)
    ]
    
    try:
        response = chat_client.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Erro ao processar chunk com chat: {e}")
        return chunk_text


def embed_with_retry(embedding_client, texts, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """
    Tenta gerar embeddings com retry em caso de erro 429 (quota excedida).
    Usa backoff exponencial.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            vectors = embedding_client.embed_documents(texts)
            
            # Verifica a dimensão dos vetores gerados
            if vectors and len(vectors) > 0:
                actual_dim = len(vectors[0])
                expected_dim = get_embedding_dimension()
                
                if actual_dim != expected_dim:
                    logger.warning(f"⚠️ Dimensão inesperada! Esperado: {expected_dim}, Obtido: {actual_dim}")
                    logger.warning("Atualizando dimensão para: {actual_dim}")
                    
                    # Atualiza a dimensão global
                    global EMBEDDING_DIM
                    EMBEDDING_DIM = actual_dim
                    # Atualiza a coluna 'embedding' para a nova dimensão
                    Document.__table__.c.embedding.type = Vector(actual_dim)
            
            return vectors
            
        except Exception as e:
            # Verifica se é erro de quota (429)
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                attempt += 1
                if attempt >= max_retries:
                    logger.error("Número máximo de tentativas atingido para embed_documents. Abortando.")
                    raise
                wait_time = initial_backoff * (2 ** (attempt - 1))
                logger.warning(f"Quota excedida (429). Aguardando {wait_time:.1f}s antes de tentar novamente (tentativa {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                logger.error(f"Erro inesperado ao gerar embeddings: {e}")
                raise
    raise RuntimeError("Falha ao gerar embeddings após múltiplas tentativas.")


def ingest_pdf():
    global EMBEDDING_DIM
    
    logger.info("=" * 60)
    logger.info("INICIANDO INGESTÃO DO PDF (MODO ECONOMIA DE TOKENS)")
    logger.info("=" * 60)
    logger.info(f"PDF_PATH = {PDF_PATH}")
    logger.info(f"Modelo de embedding: {GOOGLE_EMBEDDING_MODEL}")
    logger.info(f"CHUNK_SIZE = {CHUNK_SIZE}")
    logger.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logger.info(f"MAX_BATCHES = {MAX_BATCHES}")
    logger.info(f"SLEEP_BETWEEN_BATCHES = {SLEEP_BETWEEN_BATCHES}s")
    logger.info(f"USE_CHAT_PROCESSING = {USE_CHAT_PROCESSING}")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL)

    # Habilita extensão vector
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        logger.info("✅ Extensão vector habilitada")
    except SQLAlchemyError as exc:
        logger.error("Falha ao garantir extensão vector:", exc)
        return

    # Detecta a dimensão e recria a tabela se necessário
    if not recreate_table_with_correct_dimension(engine):
        logger.error("❌ Falha ao configurar tabela com a dimensão correta")
        return

    # Carrega o PDF
    try:
        full_text = load_pdf_text(PDF_PATH)
    except FileNotFoundError as exc:
        logger.error(exc)
        return

    if not full_text.strip():
        logger.error("O PDF está vazio ou não foi possível extrair texto.")
        return

    # Divide em chunks
    chunks = split_text(full_text)
    logger.info(f"{len(chunks)} chunks gerados a partir do PDF.")

    # Limita a quantidade de chunks
    chunks_to_process = min(len(chunks), BATCH_SIZE * MAX_BATCHES)
    chunks = chunks[:chunks_to_process]
    
    logger.info(f"⚠️  Processando APENAS os primeiros {chunks_to_process} chunks (limitado a {MAX_BATCHES} lotes de {BATCH_SIZE})")
    if len(chunks) > chunks_to_process:
        logger.info(f"⏭️  {len(chunks) - chunks_to_process} chunks restantes NÃO serão processados nesta execução")
    logger.info("=" * 60)

    # Inicializa clientes
    try:
        embeddings_client = get_embedding_client()
        chat_client = None
        if USE_CHAT_PROCESSING:
            chat_client = get_chat_client()
            logger.info(f"Chat model: {GOOGLE_CHAT_MODEL} (habilitado)")
        else:
            logger.info("Processamento com chat DESABILITADO para economizar tokens")
    except Exception as exc:
        logger.error("Erro ao inicializar clientes:", exc)
        return

    # Processamento opcional com chat
    if USE_CHAT_PROCESSING and chat_client:
        logger.info("Processando chunks com ChatGoogleGenerativeAI...")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processando chunk {i+1}/{len(chunks)}...")
            cleaned = process_chunk_with_chat(chunk, chat_client)
            processed_chunks.append(cleaned)
            time.sleep(1.0)
        chunks_to_embed = processed_chunks
    else:
        chunks_to_embed = chunks

    # Geração de embeddings
    all_vectors = []
    total_chunks = len(chunks_to_embed)
    logger.info(f"Iniciando geração de embeddings para {total_chunks} chunks...")

    total_batches = min((total_chunks + BATCH_SIZE - 1) // BATCH_SIZE, MAX_BATCHES)
    logger.info(f"Serão processados {total_batches} lotes (máximo de {MAX_BATCHES})")

    for i in range(0, min(total_chunks, BATCH_SIZE * MAX_BATCHES), BATCH_SIZE):
        batch = chunks_to_embed[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        logger.info(f"🔄 Processando lote {batch_num}/{total_batches} (chunks {i+1} a {min(i+BATCH_SIZE, total_chunks)})")

        try:
            vectors = embed_with_retry(embeddings_client, batch)
            all_vectors.extend(vectors)
            logger.info(f"✅ Lote {batch_num} concluído. {len(vectors)} embeddings gerados.")
        except Exception as exc:
            logger.error(f"❌ Falha ao gerar embeddings para o lote {batch_num}: {exc}")
            return

        if i + BATCH_SIZE < min(total_chunks, BATCH_SIZE * MAX_BATCHES):
            logger.info(f"⏳ Aguardando {SLEEP_BETWEEN_BATCHES}s antes do próximo lote...")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    logger.info(f"✅ Todos os {len(all_vectors)} embeddings gerados com sucesso.")

    # Salva no banco
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        source = os.path.basename(PDF_PATH)
        
        # Verifica a dimensão dos vetores antes de salvar
        if all_vectors:
            actual_dim = len(all_vectors[0])
            expected_dim = get_embedding_dimension()
            
            if actual_dim != expected_dim:
                logger.warning(f"⚠️ Dimensão dos vetores ({actual_dim}) diferente do esperado ({expected_dim})")
                logger.warning("Atualizando dimensão do banco para: {actual_dim}")
                EMBEDDING_DIM = actual_dim
                Document.__table__.c.embedding.type = Vector(actual_dim)
        
        for index, (chunk, vector) in enumerate(zip(chunks_to_embed[:len(all_vectors)], all_vectors)):
            session.add(
                Document(
                    source=source,
                    chunk_index=index,
                    content=chunk,
                    embedding=vector,
                )
            )
        session.commit()
        
        logger.info("=" * 60)
        logger.info(f"🎉 INGESTÃO PARCIAL CONCLUÍDA COM SUCESSO!")
        logger.info(f"📊 Processados: {len(all_vectors)} chunks de {total_chunks} totais")
        logger.info(f"📐 Dimensão do embedding: {EMBEDDING_DIM}")
        if len(chunks) > len(all_vectors):
            logger.info(f"⏭️  Restantes: {len(chunks) - len(all_vectors)} chunks não processados")
        logger.info("=" * 60)
        
    except SQLAlchemyError as exc:
        session.rollback()
        logger.error("Erro ao salvar no banco de dados:", exc)
        logger.error(f"Detalhes: {exc}")
    finally:
        session.close()


if __name__ == "__main__":
    ingest_pdf()