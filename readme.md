# 🤖 Assistente_os: RAG com Processamento Local
Este projeto foi desenvolvido para solucionar a fragmentação de informações em contextos de projetos complexos. Utilizando a arquitetura RAG (Retrieval-Augmented Generation), o assistente permite interagir com atas de reunião, documentações e evidências técnicas em formatos heterogêneos (PDF, DOCX, TXT), garantindo respostas fundamentadas e com rastreabilidade de fontes.

## 🌟 Diferenciais Técnicos
Privacidade Total: Processamento 100% local via Ollama. Nenhum dado sensível sai da infraestrutura.

Busca Semântica: Utilização de Embeddings do HuggingFace para capturar o contexto além das palavras-chave.

Rastreabilidade (Lineage): Cada resposta acompanha a citação do documento original utilizado como fonte.

Interface Intuitiva: Chat dinâmico desenvolvido em Streamlit focado na experiência do analista.

## 🏗️ Arquitetura do Sistema
Ingestão: Carregamento de documentos heterogêneos da OneDrive/Local.

Processamento: Fragmentação inteligente com RecursiveCharacterTextSplitter.

Vetorização: Geração de vetores via HuggingFaceEmbeddings.

Storage: Persistência local em banco vetorial ChromaDB.

Recuperação: Chain de recuperação customizada (LCEL) integrada ao modelo Qwen 2.5.

## 🚀 Como Executar

Pré-requisitos
Python 3.10 ou superior

Ollama instalado e rodando.

Modelo Qwen carregado: ollama pull qwen3:4b (ou a versão de sua preferência).

Instalação
Clone o repositório:

Bash

git clone https://github.com/deadpunnk/assistente_os.git
cd assistente_os
Crie e ative um ambiente virtual:

Bash

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
Instale as dependências:

Bash

pip install -r requirements.txt
Configure o arquivo .env com sua chave de API (opcional se usar apenas Ollama):

Execução
Inicie a interface do Streamlit:

Bash

streamlit run app.py
🛠️ Tecnologias Utilizadas
LangChain: Orquestração de componentes de LLM.

ChromaDB: Banco de dados vetorial de alta performance.

HuggingFace: Modelos de Embedding locais.

Ollama: Runtime para execução de LLMs open-source.

Streamlit: Framework para interfaces de dados.

📂 Estrutura de Pastas
Plaintext

├── data/               # Documentos originais (PDF, DOCX, TXT)
├── db_chroma/          # Persistência do banco vetorial (gerado automaticamente)
├── app.py              # Interface Streamlit e lógica RAG
├── requirements.txt    # Dependências do projeto
└── .env                # Variáveis de ambiente