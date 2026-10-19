# 🤖 Manual Operacional & Diretrizes de Arquitetura para Agentes de IA (AGENTS.md)

> **Versão do Projeto:** v1.0.0 Checkpoint  
> **Escopo:** Baseball ML Pitch Predictor & Statcast Tactical AI  
> **Ambiente Local:** Windows (PowerShell), Python 3.12.14 (`.venv`), Databricks Serverless, Streamlit.

Este documento é a referência técnica primária para agentes autônomos de IA e desenvolvedores que operam neste repositório. Ele sintetiza os achados da auditoria da base de código, as regras de arquitetura invioláveis, a estrutura dos componentes e as especificações para o próximo ciclo de desenvolvimento (integração com Visão Computacional).

---

## 1. Estado Atual da Base de Código (Checkpoint v1.0.0)

Após análise aprofundada da base de código, o projeto encontra-se em estado funcional, testado e modular:
- **Suíte de Testes**: **133 testes automatizados** passando com 100% de sucesso (`pytest 9.1.1` em ~60 segundos).
- **Ambiente de Execução**: Python virtual environment localizado em `.\.venv\Scripts\python.exe`.
- **Camada de Apresentação (Streamlit)**: Arquitetura multipágina moderna com 4 módulos operacionais:
  1. `pages/00_home_apresentacao.py`: Apresentação executiva e amigável para leigos com 4 perfis de arremessador e 3 momentos táticos.
  2. `pages/01_live_predictor.py`: Simulador tático lance a lance com controle de contagem, corredores em base e inferência em tempo real.
  3. `pages/02_scout_report.py`: Scouting report completo com análise de tendências na matriz 4x3 de 12 contagens e plano de jogo.
  4. `pages/03_minigame.py`: Minigame retrô arcade (8-bit / 16-bit) com máquina de estados, sistema de atributos RPG (100 pontos), sistema de vidas, multiplicador de combo e áudio sintetizado Chiptune.
- **Camada de Inferência Resiliente**: `PitchServingClient` com suporte duplo:
  - Chamadas autenticadas ao endpoint Databricks Model Serving via Databricks SDK / REST API.
  - Modo Mock Offline inteligente (gera probabilidades táticas baseadas nos priors históricos e na contagem caso o endpoint esteja inacessível ou o usuário não possua credenciais ativas).
- **Componentes Visuais Puros**: Geradores de SVG programáticos em `ui_components.py` para o Campo de Beisebol com corredores dinâmicos e Zona de Strike Statcast 3x3 com zonas periféricas de chase (11 a 14).

---

## 2. Invariantes de Arquitetura Invioláveis

Todo agente atuando neste projeto **DEVE SEGUIR ESTRITAMENTE** as 4 invariantes abaixo:

### Invariante 1: Reprodutibilidade de Ambiente e Unity Catalog (IaC Invariant)
- **Proibido Comandos Ad-Hoc no Terminal**: Nunca crie catálogos, esquemas ou volumes executando comandos manuais. Toda infraestrutura deve ser declarada como código para manter paridade entre Dev, Staging e Produção.
- **Artefatos de Infraestrutura Obrigatórios**:
  - `scripts/setup_unity_catalog.sql`: DDL idempotente com tokens `${catalog}`, `${schema}` e `${volume}`.
  - `scripts/setup_unity_catalog.py`: CLI de automação com tratamento de privilégios e fallback para usuários sem permissão de `CREATE CATALOG`.
  - `notebooks/00_setup_environment.py`: Tarefa inicial obrigatória de qualquer Lakeflow Job.
- **Parametrização**: Recursos em `resources/*.yml` e `databricks.yml` devem referenciar `${var.catalog}`, `${var.schema}` e `${var.volume}`.

### Invariante 2: Lakeflow Spark Declarative Pipelines (@dp) em Serverless
- **Resolução Dinâmica de `sys.path`**: Em pipelines Serverless, o driver container não inclui o diretório `src/` no `sys.path`. Todo arquivo de pipeline declarativo que importa módulos locais deve resolver dinamicamente os caminhos candidatos (`/Workspace/Users/...`, `/Workspace/...` e diretórios pais de `__file__`).
- **Definições de Fallback Seguras**: Fornecer constantes de fallback para chaves primárias e regras de qualidade para evitar que a descoberta do grafo Spark falhe (`ModuleNotFoundError`).
- **Edição ADVANCED para Auto CDC**: Pipelines que usam `dp.create_auto_cdc_flow` exigem `edition: ADVANCED`.
- **Configuração de Iteração em Dev**: Manter `channel: PREVIEW`, `development: true` e `continuous: false`.

### Invariante 3: Engenharia de Features Pré-Arremesso (Zero-Leakage Invariant)
- **Isolamento de Sequência Intra-At-Bat**: As features do arremesso anterior (`prev_pitch_*`) devem ser particionadas estritamente por `(game_pk, at_bat_number)` e ordenadas por `pitch_number`. O primeiro arremesso do duelo (`pitch_number == 1`) deve receber deterministamente `'First Pitch'` / `0`.
- **Priors do Arremessador Estritamente Passados**: Métricas históricas de tendência do arremessador devem utilizar janela particionada por `pitcher`, ordenada cronologicamente por `(game_date, game_pk, at_bat_number, pitch_number)` com limites `rowsBetween(Window.unboundedPreceding, -1)` e suavização Laplaciana/Bayesiana com prior da liga, garantindo zero vazamento de dados futuros.
- **Filtragem Silver**: Falhas de sensores e arremessos atípicos (Knuckleball, Eephus, Pitchout) devem ser descartados na camada Silver.

### Invariante 4: Fluxo de Desenvolvimento TDD Multagente
1. **Red Agent (QA)**: Escreve testes unitários e de integração em `tests/test_*.py` cobrindo casos limites e comportamentos esperados (fase de falha inicial).
2. **Green Agent (Developer)**: Implementa o código mínimo e limpo em `src/` para fazer 100% dos testes passarem.
3. **Gatekeeper (Reviewer)**: Executa a suíte completa de testes, verifica linter e audita o `git diff` antes de validar a entrega.

---

## 3. Mapeamento de Diretórios & Responsabilidades

```text
pitch_predictor/
├── config/
│   └── config.yaml               # Configuração padrão de catálogo, esquema, volume e endpoint
├── notebooks/
│   ├── 00_setup_environment.py   # Setup idempotente do metastore/schema no Unity Catalog
│   ├── 01_data_ingestion.py      # Download Statcast via PyBaseball e carga Delta Bronze
│   ├── 02_automl_run.py          # Disparo do Databricks AutoML com split temporal
│   └── 03_model_inference.py     # Inferência em lote e validação de métricas MLflow
├── pages/
│   ├── 00_home_apresentacao.py   # Apresentação descomplicada com perfis de arremessador
│   ├── 01_live_predictor.py      # Simulador tático pitch-by-pitch com controle de contagem
│   ├── 02_scout_report.py        # Scouting report detalhado com matriz 4x3 de 12 contagens
│   └── 03_minigame.py            # Minigame retrô arcade (Você vs. IA no Montinho)
├── resources/
│   ├── pitch_predictor_app.yml   # Definição do Databricks App
│   ├── pitch_predictor_endpoint.yml # Definição do Endpoint de Model Serving Serverless
│   ├── pitch_predictor_job.yml   # Definição do Job com 3 tarefas orquestradas
│   └── pitch_predictor_pipeline.pipeline.yml # Definição do Pipeline Declarativo Lakeflow
├── scripts/
│   ├── setup_unity_catalog.py    # Utilitário CLI para inicialização do Unity Catalog
│   ├── setup_unity_catalog.sql   # DDL SQL idempotente com tokens de catálogo/esquema
│   ├── verify_browser_dom.py     # Script de inspeção de DOM das páginas Streamlit
│   ├── verify_minigame_browser.py# Teste E2E headless do minigame arcade via Playwright
│   └── verify_ui_server.py       # Probe de saúde HTTP da UI
├── src/pitch_predictor/
│   ├── config.py                 # Resolução de variáveis e cliente WorkspaceClient
│   ├── features.py               # Colunas Silver, mapeamentos e transformações zero-leakage
│   ├── ingestion.py              # Download de dados Statcast e sanitização inicial
│   ├── minigame_engine.py        # Motor de jogo puro (regras, máquina de estados, resolução de duelos)
│   ├── pipelines/bronze_pitches.py # Pipeline Lakeflow DLT para camada Bronze
│   ├── scenarios.py              # Dicionários de cenários e perfis táticos
│   ├── scout_analytics.py        # Agregações analíticas e cálculo da matriz de contagens
│   ├── serving_client.py         # Cliente de inferência (Databricks Serving com Mock fallback)
│   ├── training.py               # Orquestração do AutoML e rastreamento de experimentos MLflow
│   ├── ui_components.py          # Componentes visuais SVG (Campo de Beisebol e Strike Zone)
│   └── utils.py                  # Logging formatado e sessão Spark local
└── tests/                        # 133 testes automatizados (pytest)
```

---

## 4. Auditoria do Projeto de Visão Computacional (`visao_computacional`)

Foi realizada uma auditoria detalhada no projeto irmão localizado em [`d:\PI\visao_computacional`](file:///d:/PI/visao_computacional):
- **Objetivo**: Aplicação de reconhecimento facial de emoções em tempo real via webcam com espelho de emojis e minigame de mímica.
- **Stack Técnica**: Python 3.12, OpenCV (`opencv-python`), FER (*Facial Emotion Recognition*), TensorFlow Lite, NumPy e Pillow.
- **Classificações**: 7 emoções básicas (`happy`, `sad`, `angry`, `surprise`, `fear`, `disgust`, `neutral`).
- **Mecanismos Existentes**:
  - `EmotionDetector` (`src/detector.py`): Detecção facial via Haar Cascade e classificação por CNN em TFLite.
  - `EmojiOverlay` (`src/overlay.py`): Composição digital por alpha blending de máscaras e cards HUD translúcidos.
  - `EmojiGame` (`src/game.py`): Máquina de estados com rodadas temporizadas, cálculo de streak e bônus de velocidade.
  - Performance: Otimizado para 30+ FPS com salto de frames (*frame skip*) e suavização temporal para evitar oscilações.

---

## 5. Especificações para a Próxima Fase: Integração de CV no Pitch Predictor

### 5.1 Diretrizes de Isolamento de Dependências
- As dependências de Visão Computacional (`opencv-python`, `fer`, `tensorflow`, `Pillow`) **DEVEM ser opcionais** no `pyproject.toml` (sob o grupo `vision`).
- **Motivo**: Os jobs batch do Databricks Serverless e o Lakeflow Pipeline **não devem instalar OpenCV ou drivers de vídeo**, preservando o tempo de inicialização e a estabilidade.

### 5.2 Arquitetura do Novo Subpacote `src/pitch_predictor/vision/`
1. `__init__.py`: Importações seguras com verificação de disponibilidade (`HAS_VISION = True/False`).
2. `detector.py`: Adaptador de detecção com **Modo Headless / Mock Frame** obrigatório para execução de testes em ambientes de CI/CD sem webcam.
3. `composure.py`: Motor de biofeedback tático que traduz emoções em métricas do beisebol:
   - **Índice de Compostura (0.0 a 1.0)**: Neutralidade e foco geram bônus de controle e precisão.
   - **Fator de Tilt (0.0 a 1.0)**: Medo, raiva ou estresse indicam descontrole do arremessador sob pressão.
4. `overlay.py`: HUD estilizado no padrão Statcast (neon cyan `#00FFCC` e neon pink `#FF007F`) para sobreposição na captura de vídeo.

### 5.3 Integração com o Minigame (`pages/03_minigame.py` e `minigame_engine.py`)
- O método `resolve_arcade_pitch` receberá parâmetro opcional `composure_score: float = 1.0` (garantindo retrocompatibilidade total com todos os 133 testes existentes).
- Na interface do minigame, a "Face-Cam Tática" será uma funcionalidade opcional (toggle).

### 5.4 Nova Página Dedicada (`pages/04_visao_computacional.py`)
- Módulo Streamlit dedicado para:
  1. Análise biométrica de compostura facial em tempo real.
  2. Upload de vídeos de arremessos para análise de reações emocionais.
  3. Demonstração de calibração geométrica de Zona de Strike sobre vídeo.

---

## 6. Comandos Operacionais para Agentes de IA

### Execução de Testes
```powershell
# Executar toda a suíte de testes (obrigatório após qualquer alteração)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Executar um teste específico
.\.venv\Scripts\python.exe -m pytest tests/test_minigame_engine.py -v
```

### Verificação de Linting e Formatação
```powershell
# Verificação com ruff
.\.venv\Scripts\python.exe -m ruff check src/ tests/ pages/
```

### Execução da Aplicação Localmente
```powershell
# Iniciar servidor Streamlit
.\.venv\Scripts\streamlit.exe run app.py
```

### Validação do Bundle Databricks
```bash
databricks bundle validate -p dev
```

---
*Este documento deve ser mantido atualizado a cada novo checkpoint de versão do projeto.*
