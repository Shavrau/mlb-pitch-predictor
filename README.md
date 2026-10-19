# ⚾ MLB Pitch Predictor — Statcast Tactical AI (v1.0.0 Checkpoint)

> **Plataforma Modular de Inteligência Artificial para Predição Tática de Arremessos de Beisebol**  
> Desenvolvida com **Databricks Serverless**, **AutoML**, **Lakeflow Spark Declarative Pipelines**, **MLflow Model Serving** e interface interativa multipágina em **Streamlit**.

---

## 📌 Visão Geral do Projeto

O **MLB Pitch Predictor** é um sistema preditivo que analisa a dinâmica duelo a duelo entre arremessador (*pitcher*) e rebatedor (*batter*) na Major League Baseball (MLB). Utilizando telemetria física e contextual do **Statcast**, o modelo prevê em tempo real qual categoria tática de arremesso será lançada:
- 🚀 **Fastball** (Bola Rápida — *Four-Seam, Sinker, Cutter*)
- 🌪️ **Breaking Ball** (Efeito Pronunciado — *Slider, Sweeper, Curveball, Knuckle Curve*)
- 🛑 **Offspeed** (Variação de Velocidade & Ação Tardia — *Changeup, Splitter*)

### Pilares Arquiteturais
1. **Engenharia de Features Zero-Leakage**: Janelas temporais estritamente passadas e isoladas por confronto (*intra-at-bat*), eliminando qualquer risco de vazamento de dados futuros (*data leakage*).
2. **Infraestrutura como Código (IaC Invariant)**: Definição declarativa e idempotente de Catalogs, Schemas e Volumes no Unity Catalog via Databricks Asset Bundles (DABs).
3. **Pipeline Declarativo Serverless**: Ingestão e transformação Bronze/Silver utilizando Lakeflow Spark Declarative Pipelines (`@dp`).
4. **Interface Multimodal & Gamificada**: Painel tático profissional estilo transmissão esportiva e minigame retrô arcade (8-bit / 16-bit) para duelo direto entre o usuário e o modelo preditivo.
5. **Confiabilidade & Qualidade de Código**: Suíte abrangente de 133 testes automatizados (100% de aprovação) executada localmente e preparada para CI/CD.

---

## 🗂️ Estrutura do Repositório

```text
pitch_predictor/
├── .agents/                        # Habilidades, diretrizes e regras para agentes de IA
├── config/
│   └── config.yaml                 # Configuração de catálogo, esquema, volume e endpoint
├── notebooks/
│   ├── 00_setup_environment.py     # Inicialização idempotente do catálogo/esquema no Unity Catalog
│   ├── 01_data_ingestion.py        # Coleta Statcast via PyBaseball e carga no Delta Lake (Bronze)
│   ├── 02_automl_run.py            # Execução de AutoML do Databricks com otimização bayesiana
│   └── 03_model_inference.py       # Avaliação e registro de modelos no MLflow / Unity Catalog
├── pages/
│   ├── 00_home_apresentacao.py     # Visão Geral & Apresentação Descomplicada (Live Demo para leigos)
│   ├── 01_live_predictor.py        # Simulador Tático em Tempo Real & Pitch-by-Pitch Tracker
│   ├── 02_scout_report.py          # MLB Scouting Report, Matriz 4x3 de 12 Contagens & Game Plan
│   └── 03_minigame.py              # Minigame Retrô Arcade: Você vs. IA no Montinho (8-bit / Chiptune)
├── resources/
│   ├── pitch_predictor_app.yml     # Definição do Databricks App resource
│   ├── pitch_predictor_endpoint.yml# Definição do Databricks Model Serving Endpoint
│   ├── pitch_predictor_job.yml     # Workflow orquestrado Serverless (Setup -> Ingestão -> AutoML)
│   └── pitch_predictor_pipeline.pipeline.yml # Lakeflow Spark Declarative Pipeline definition
├── scripts/
│   ├── setup_unity_catalog.py      # Script CLI para configuração idempotente do Unity Catalog
│   ├── setup_unity_catalog.sql     # DDL idempotente parametrizado (${catalog}, ${schema})
│   ├── verify_browser_dom.py       # Verificação automatizada de DOM para páginas Streamlit
│   ├── verify_minigame_browser.py  # Teste E2E do Minigame Arcade via Playwright/Browser
│   └── verify_ui_server.py         # Probe de saúde HTTP para o servidor Streamlit
├── src/
│   └── pitch_predictor/            # Pacote Python principal reutilizável
│       ├── __init__.py             # Inicialização do pacote
│       ├── config.py               # Resolução de configurações e WorkspaceClient Databricks
│       ├── features.py             # Definição de colunas Silver, invariantes e transformações
│       ├── ingestion.py            # Coleta de dados Statcast via PyBaseball com sanitização
│       ├── minigame_engine.py      # Motor de regras do minigame arcade (RPG stats, resolução de duelos)
│       ├── pipelines/              # Lakeflow pipelines (@dp)
│       │   └── bronze_pitches.py   # Ingestão Delta Live Tables / Lakeflow Serverless
│       ├── scenarios.py            # Cenários de jogo pré-configurados e perfis de arremessadores
│       ├── scout_analytics.py      # Agregações e matrizes estatísticas de scouting report
│       ├── serving_client.py       # Cliente resiliente de inferência (Databricks Serving + Mock Offline)
│       ├── training.py             # Utilitários de orquestração do AutoML e rastreamento MLflow
│       ├── ui_components.py        # Renderizadores SVG puros (Campo de Beisebol, Strike Zone 3x3)
│       └── utils.py                # Utilitários de logging e Spark Session
├── tests/                          # Suíte de 133 testes automatizados (pytest)
│   ├── test_automl_training.py     # Testes de orquestração do AutoML e splits temporais
│   ├── test_config.py              # Testes de resolução de variáveis e catálogo
│   ├── test_ingestion.py           # Testes de validação de schemas Statcast e limites
│   ├── test_minigame_engine.py     # Testes da máquina de estados do jogo e matriz de resolução
│   ├── test_scout_analytics.py     # Testes da matriz 4x3 de contagens e agregações
│   ├── test_serving_client.py      # Testes de inferência com endpoint e fallback de mock
│   ├── test_silver_features.py     # Testes de isolamento intra-at-bat e zero leakage
│   ├── test_streamlit_app.py       # Testes headless de navegação e renderização das páginas
│   └── test_ui_components.py       # Testes de integridade de SVGs gerados programaticamente
├── AGENTS.md                       # Diretrizes, invariantes e manual operacional para Agentes de IA
├── app.py                          # Entrypoint principal da aplicação multipágina Streamlit
├── app.yaml                        # Metadados de empacotamento do Databricks App
├── databricks.yml                  # Configuração do Databricks Asset Bundle (DAB)
├── pyproject.toml                  # Especificações de empacotamento Python
├── requirements.txt                # Dependências fixadas do projeto
└── sample_inference_request.json   # Payload de exemplo para testes de inferência no Model Serving
```

---

## 🚀 Módulos da Aplicação Interativa

A aplicação web é executada via Streamlit e possui 4 módulos principais:

### 1. 🏠 Visão Geral & Apresentação Descomplicada (`pages/00_home_apresentacao.py`)
- Desenvolvida para demonstrações rápidas para qualquer audiência (sem necessidade de conhecimento técnico prévio).
- Permite selecionar perfis amigáveis de arremessador (*Lança-Chamas*, *Mestre dos Efeitos*, *Mudança & Tempo*) e situações de jogo (*0-0*, *3-0 Encurralado*, *0-2 Pressão Total*).
- Executa a inferência instantânea exibindo a probabilidade visual com gauges e explicação tática clara.

### 2. ⚾ Simulador Tático Ao Vivo (`pages/01_live_predictor.py`)
- Simulador completo lance a lance (*pitch-by-pitch tracker*).
- Controle de contagem (bolas/strikes), situação de bases (1ª, 2ª, 3ª), entradas, placar e histórico do duelo.
- Consulta de arremessadores reais da MLB com priors calculados dinamicamente via Databricks ou base local.

### 3. 📋 MLB Scouting Report & Matriz 4x3 (`pages/02_scout_report.py`)
- Dossiê detalhado de tendências do arremessador ao longo de todas as 12 contagens possíveis do beisebol (`0-0` a `3-2`).
- Identificação automática do "Arremesso de Confiança" (*Go-To Pitch*) e "Arremesso de Eliminação" (*Put-Away Pitch*).
- Game plan ofensivo automatizado para guiar a abordagem do rebatedor.

### 4. 🎮 Minigame Retrô Arcade: Você vs. IA (`pages/03_minigame.py`)
- Estética retrô fliperama dos anos 90 (8-bit / 16-bit com fontes *Press Start 2P* e cores Neon).
- Distribuição de 100 pontos RPG de atributos (Velocidade, Efeito, Variação).
- Mecânica de sobrevivência infinita com vidas, sequência de strikeouts, multiplicador de combo e High Score.
- Efeitos sonoros sintetizados em tempo real via Web Audio API do navegador.

---

## 🛠️ Execução Local e Testes

### 1. Configurar Ambiente Virtual
```powershell
# Criação e ativação do ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalação das dependências e do pacote local
pip install -r requirements.txt
pip install -e .
```

### 2. Executar a Suíte de Testes (133 testes)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 3. Iniciar a Aplicação Streamlit
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Acesse a aplicação no navegador em: `http://localhost:8501`

---

## ☁️ Deploy e Operação no Databricks (DABs)

### 1. Autenticação e Validação
```bash
# Autenticar com o workspace Databricks
databricks auth login --profile dev

# Validar o Databricks Asset Bundle
databricks bundle validate -p dev
```

### 2. Deploy dos Recursos
```bash
# Fazer o deploy dos notebooks, pipelines, jobs e apps
databricks bundle deploy -p dev
```

### 3. Disparo do Workflow
```bash
# Executar o workflow orquestrado no Databricks Serverless
databricks bundle run pitch_predictor_job -p dev
```

---

## 🔮 Próximos Passos (Roadmap)

- **Integração com Visão Computacional (CV)**:
  - Incorporação do motor de reconhecimento facial de emoções (`d:\PI\visao_computacional`).
  - Câmera tática de compostura no Minigame Arcade: cálculo do índice *Poker Face* do arremessador sob pressão.
  - Laboratório de Visão Computacional (`pages/04_visao_computacional.py`) para análise de compostura e demonstração de zona de strike ótica.
