# Desafio MBA Engenharia de Software com IA - Full Cycle

Esta aplicação faz ingestão de um PDF, gera embeddings e armazena os chunks em um banco PostgreSQL com extensão pgvector para permitir busca semântica e perguntas sobre o conteúdo carregado.

## Passo a passo para executar

### 1. Pré-requisitos

- Docker Desktop instalado e em execução
- Python 3.10+ instalado
- Uma chave da API do Google Gemini configurada
- Um arquivo PDF para ser indexado

### 2. Criar o arquivo .env

Na raiz do projeto, crie um arquivo chamado .env com as variáveis abaixo:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
GOOGLE_API_KEY=sua_chave_google
PDF_PATH=document.pdf
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_CHAT_MODEL=gemini-1.5-flash
CHUNK_SIZE=100
CHUNK_OVERLAP=20
EMBEDDING_BATCH_SIZE=3
```

> O arquivo PDF deve estar na raiz do projeto ou no caminho informado em PDF_PATH.

### 3. Subir o banco com Docker Compose

No diretório do projeto, execute:

```bash
docker compose up -d
```

Esse comando sobe o container do PostgreSQL com a extensão pgvector. O serviço de bootstrap garante que a extensão seja criada automaticamente.

Para verificar se o banco está pronto:

```bash
docker compose ps
```

### 4. Instalar as dependências Python

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Em seguida, instale as bibliotecas:

```bash
pip install -r requirements.txt
```

### 5. Executar a ingestão do PDF

Com o banco já rodando, execute o script de ingestão:

```bash
python src/ingest.py
```

Esse processo irá:
- ler o PDF definido em PDF_PATH;
- dividir o texto em chunks;
- gerar embeddings com o modelo configurado;
- salvar os dados na tabela documents do PostgreSQL.

### 6. Consultar os dados ingeridos (opcional)

Para visualizar os registros armazenados no banco:

```bash
python src/querydb.py
```

### 7. Usar o chat de busca

Depois da ingestão, inicie o chat:

```bash
python src/chat.py
```

O programa irá abrir um loop de perguntas. Digite uma pergunta relacionada ao conteúdo do PDF e o sistema responderá com base no contexto recuperado do banco.

Exemplo:

```text
Pergunta> O que este documento fala sobre?
```

### 8. Encerrar tudo

Para parar os containers do Docker:

```bash
docker compose down
```

Se quiser remover também os dados persistidos do banco:

```bash
docker compose down -v
```