# 🤖 Manual Operacional & Diretrizes de Arquitetura para Agentes de IA (AGENTS.md)

> **Versão do Projeto:** v2.2.0 Checkpoint (Sprint 1: Anatomical Pose Heatmap & Retro Arcade Hotkeys)  
> **Escopo:** Baseball ML Pitch Predictor, Statcast Tactical AI, Biomechanical Pose Estimation, Multimodal Fusion & Databricks Apps Serverless Cloud  
> **Ambiente Local:** Windows (PowerShell), Python 3.12.14 (`.venv`), Databricks Serverless, Streamlit 1.66.1.

Este documento é a referência técnica primária para agentes autônomos de IA e desenvolvedores que operam neste repositório. Ele sintetiza os achados da auditoria da base de código, as regras de arquitetura invioláveis, a estrutura dos componentes e as especificações para o próximo ciclo de desenvolvimento.

---

## 1. Estado Atual da Base de Código (Checkpoint v2.1.0)

Após homologação e implementação das correções de contraste e usabilidade Dual-Mode:
- **Suíte de Testes**: **227 testes automatizados** passando com 100% de sucesso (`pytest 9.1.1`).
- **Entregáveis da Sprint 1 Homologados**:
  - **Heatmap Anatômico Interativo no Frame de Soltura 3D (`src/pitch_predictor/vision/overlay.py` e `pages/04_pitch_vision_lab.py`)**:
    - Código de cores dinâmico nos nós anatômicos e vetores cinemáticos calibrado pelo Z-score intra-atleta:
      - `🟢 Emerald / Cyan (#22D4E8)`: Mecânica normal na baseline do arremessador ($|Z| < 1.0$).
      - `🟡 Amber (#F59E0B)`: Microdesvio articular moderado sob atenção ($1.0 \le |Z| < 1.75$).
      - `🔴 Ruby Red (#EF4444)`: Desvio biomecânico estatisticamente anômalo / indício de Pitch Tipping ($|Z| \ge 1.75$).
    - Cache em memória dos marcos anatômicos do frame base (`_BASE_FRAME_LANDMARKS_CACHE`), garantindo re-renderização ultrarrápida (<1ms) durante a manipulação dos sliders interativos no simulador de pose.
    - Faixa editorial de legenda sob o visualizador 3D com badges de status em ambos os temas (Dark Broadcast e Light Editorial).
  - **Atalhos de Teclado no Minigame Arcade (`pages/03_minigame.py` e `src/pitch_predictor/ccv2_components.py`)**:
    - Teclas `1`, `2`, `3`: Alternância instantânea de arsenal entre Tier S, Tier A e Tier B.
    - Teclas Direcionais (`Setas ↑ ← ↓ →` e `W, A, S, D`): Navegação bidirecional contínua da mira entre as 13 zonas (grid 3x3 + 4 quadrantes de chase) governada por `NAV_MAP` nos componentes Streamlit CCv2 e ouvinte DOM global.
    - Teclas `Espaço` ou `Enter`: Disparo instantâneo do arremesso contra a IA (`ARREMESSAR CONTRA A IA`).
    - Faixa visual de controles arcade estilo console retrô no HUD Scorebug com teclas `<kbd>` estilizadas de alto contraste WCAG AAA.
- **Homologação Visual & Contraste WCAG AAA**:
  - **Erradicação do Fundo Branco no Card da Strike Zone CCv2 (`pages/03_minigame.py` e `src/pitch_predictor/ccv2_components.py`)**: Aplicação de tokens explícitos `.pp-sz-wrap:not(.pp-light)` (`#12141A` / `#222631`) e `.pp-sz-wrap.pp-light` (`#F1F5F9` / `#C2CFDD`), transmitindo `is_light` no payload `data` e adaptando as zonas interativas do grid para total harmonia em ambos os temas.
  - **Legibilidade Aperfeiçoada da Seção 3 no Modo Light Editorial (`pages/04_pitch_vision_lab.py`)**: Delta de probabilidade calibrada com badge verde esmeralda escuro (`#064E3B` / `#065F46`), banner editorial de confirmação biomecânica com fundo `#ECFDF5`, borda `#059669` e texto `#064E3B`, e pills de auditoria Out-of-Fold em cinza grafite de alto contraste sobre `#E2E8F0`.
  - **Contraste dos Sliders de Deformação e Métricas Cinemáticas (`pages/04_pitch_vision_lab.py` e `src/pitch_predictor/ui_components.py`)**: Deltas dinâmicos padronizados com badges de alto contraste (esmeralda `#064E3B` para deltas positivos e bordô `#991B1B` para negativos em Light Mode) e rótulos de sliders em `#334155`.
  - **Cards Interativos de Navegação Rápida no Rodapé da Home (`pages/00_home_apresentacao.py`)**: Substituição de links simples por 3 cards editoriais contendo kicker, ícone, descrição das capacidades e botão direto para Minigame, Scouting Report e Laboratório de Visão.
- **Ambiente de Execução**: Python virtual environment localizado em `.\.venv\Scripts\python.exe` com `opencv-python-headless`, `mediapipe`, `shap`, `mlflow`, `scipy` e `databricks-sdk` instalados.
- **Aplicação em Nuvem Operacional (Databricks Apps)**:
  - App `pitch-predictor-app-dev` em estado **`RUNNING` / `ACTIVE`**.
  - URL Corporativa Pública: **`https://pitch-predictor-app-dev-7474645845452998.aws.databricksapps.com`**.
  - Deploy **`01f1d481d17e1d048a2fbcbaad01611d`** ativo com **`SUCCEEDED`** e Streamlit `1.66.1`.
- **Revisão Geral de Escrita & Redação Esportiva**:
  - **Correções Gramaticais e Semânticas**: Correção da regência preposicional diante de sujeitos no infinitivo (*"antes de ele acontecer"*, *"tempo de o olho processar"*, *"antes de a bola deixar sua mão"*).
  - **Acentuação e Termos Técnicos**: Inclusão de acentos em relatórios gerados por modelos (*"afastando a hipótese de"*), substituição de termos de pôquer por esportivos (*"Sinais Físicos"* em vez de *"Tells Físicos"*).
  - **Consistência de Navegação e Botões**: Remoção de emojis duplicados nos links de navegação rápida da Home e aprimoramento de clareza textual nos cards e painéis.
  - **Terminologia Canônica Preservada**: Termos universais de transmissão esportiva (*Groundout*, *Groundballs*, *Wipeout*, *Platoon*, *Chase*, *Whiff*, *Double Play*) mantidos integralmente, mantendo a erradicação do termo artificial *"tunelamento"* em favor de *"🎭 Disfarce de Trajetória"*.
- **Laboratório de Visão & Biomecânica Dinâmico (`pages/04_pitch_vision_lab.py`)**:
  - **Seletor de Arremessadores Oficiais da MLB (`st.selectbox(key="vision_pitcher_selector")`)**:
    - Suporte completo aos 5 astros: Paul Skenes (RHP), Corbin Burnes (RHP), Logan Webb (RHP), Chris Sale (LHP), Yoshinobu Yamamoto (RHP).
    - Sincronização automática de manifestos, clipes de vídeo, datasets biomecânicos e snapshots calibrados de liberação (Release Point).
    - Detecção automática de lateralidade (Destro vs. Canhoto): adaptação do esqueleto anatômico no momento de soltura (braço esquerdo dominante para Chris Sale).
    - Sincronização dos relatórios SHAP e dicas táticas customizadas para o arremessador selecionado.
- **Minigame Arcade - Seleção em Cards Verticais Lado a Lado (`pages/03_minigame.py` e `src/pitch_predictor/ui_components.py`)**:
  - **Layout de 5 Cards Verticais (`st.columns(5)`)**:
    - Disposição lado a lado dos 5 astros da MLB (Skenes, Burnes, Webb, Sale, Yamamoto).
    - **Topo do Card**: Tags de conquistas e qualidades em destaque (`[🏆 Cy Young]`, `[🚀 102 mph]`, `[🌪️ 2800+ RPM]`, `[🏹 Canhoto LHP]`, `[👻 Bola Fantasma]`, etc.), nome do jogador e franquia.
    - **Meio do Card**: Fotos reais oficiais de estúdio dos 5 astros da MLB (`render_pitcher_photo_html`), incorporadas em alta resolução com fundo transparente via Base64 local (`src/pitch_predictor/assets/headshots/`) com fallback para o CDN da MLB (`img.mlbstatic.com`), dentro de uma moldura neon arcade retro.
    - **Parte de Baixo do Card**: Características de arsenal organizadas em Tiers (Tier S, Tier A, Tier B com velocidades em mph), resumo tático e botão direto de seleção `🎯 Escolher [Nome]`.
- **Motor Probabilístico Calibrado por Tiers**:
  - O rebatedor não obtém automaticamente uma rebatida mesmo se a IA adivinhar o arremesso.
  - Arremessos de Tier S possuem alta probabilidade de induzir foul ball (strike) ou contato fraco (groundout/flyout -> Out).
- **Medidor de Timing do Rebatedor (Sistema Anti-Spam)**:
  - Arremessar o mesmo tipo de bola consecutivamente eleva o medidor de calor do rebatedor (`🟩 Frio` $\rightarrow$ `🟨 Alerta` $\rightarrow$ `🟥 Na Espera!`).
  - Com o medidor alto ($\ge 2$), se a IA adivinhar o arremesso, resulta em punição severa (Home Run / rebatida extra-base).
  - Variar o arremesso para Tier A ou B quebra o timing do rebatedor ("🧊 Quebrou o timing!") e resfria o medidor imediatamente a zero.
- **Astros Reais da MLB no Bastão (`MLB_BATTERS`)**:
  - Shohei Ohtani (Power Slugger, Anti-Fastball), Aaron Judge (Power Giant, High Zone Punisher), Juan Soto (Eye of the Hawk / Plate Discipline), Mookie Betts (Contact King), Bryce Harper (Aggressive Ambush).
- **Aplicação em Nuvem Operacional (Databricks Apps)**:
  - App `pitch-predictor-app-dev` em estado **`RUNNING` / `ACTIVE`**.
  - URL Corporativa Pública: **`https://pitch-predictor-app-dev-7474645845452998.aws.databricksapps.com`**.
  - Deploy `01f1ce2863da10c4a7dec107cb5c6ebd` ativo com `SUCCEEDED`.
  - URL Corporativa Pública: **`https://pitch-predictor-app-dev-7474645845452998.aws.databricksapps.com`**.
  - Deploy `01f1ce273e601e079634999dc8d78299` ativo com `SUCCEEDED`.
- **Centralização em `MLB_PITCHERS` (Single Source of Truth)**:
  - Registro único estruturado em `src/pitch_predictor/scenarios.py` com priors reais, arsenal, mão de arremesso (R/L) e perfil biomecânico calibrado para os 5 astros:
    1. **Paul Skenes** (`#694973` - Pittsburgh Pirates, RHP, Power Pitcher, 102 mph Fastball / Splinker)
    2. **Corbin Burnes** (`#669203` - Baltimore Orioles, RHP, Breaking Specialist, Cutter / Curveball)
    3. **Logan Webb** (`#657277` - San Francisco Giants, RHP, Offspeed Master, Sinker / Changeup)
    4. **Chris Sale** (`#519242` - Atlanta Braves, LHP, Lefty Crafty Ace, Low 3/4 Slot / Sweeping Slider)
    5. **Yoshinobu Yamamoto** (`#808967` - Los Angeles Dodgers, RHP, Devastating Splitter Ace, Rainbow Curve / 96 mph FF)
  - Funções utilitárias canônicas: `get_pitcher_profile()` e `get_pitcher_names()`.
- **Motor Unificado de Fallback Offline (`src/pitch_predictor/fallback_engine.py`)**:
  - Eliminação de fallbacks concorrentes e discrepantes:
    - `predict_tactical_fallback()`: Inferência tática situacional calibrada por contagem, vantagem, platoon e RISP.
    - `predict_multimodal_fallback()`: Fusão ponderada de Statcast e cinemática corporal com diagnóstico de tipping.
    - `simulate_12_counts_fallback()`: Geração determinística da matriz de 12 contagens para o Scouting Report.
- **Simplificação da UI & Remoção da Página 01 (`01_live_predictor.py`)**:
  - Eliminação da página `pages/01_live_predictor.py` para erradicar sobrecarga cognitiva (18 controles simultâneos) e redundância.
  - A aplicação passa a ter **4 páginas focadas e de alto valor**:
    1. `pages/00_home_apresentacao.py`: Visão geral executiva com seleção dos 5 arremessadores oficiais e rodapé com links para os 3 submódulos.
    2. `pages/03_minigame.py`: Minigame arcade retrô (zero-scroll) com presets oficiais e resolução via fallback unificado.
    3. `pages/02_scout_report.py`: Scouting report completo com matriz 4x3 de 12 contagens e plano de jogo.
    4. `pages/04_pitch_vision_lab.py`: Laboratório biomecânico com player de vídeo, anotação esquelética no frame de soltura e estudo de ablação.
- **Seleção Dinâmica de Arremessador na UI**:
  - `pages/04_pitch_vision_lab.py`: Seletor de arremessador no topo, sincronizando clipes, ângulos anatômicos, esqueleto (com suporte destros/canhotos), relatórios de tipping e estudo de ablação com o jogador selecionado.
  - `pages/01_live_predictor.py`: Seletor de arremessador oficial MLB, aplicando priors e lateralidade (p_throws) ao simulador tático.
  - `pages/02_scout_report.py`: Scouting report adaptado para arremessadores específicos e arquétipos gerais.
  - `pages/00_home_apresentacao.py`: Cards dos 5 arremessadores estrela da MLB.
- **Lakeflow Job Serverless End-to-End Totalmente Operacional**:
  - Job `pitch_predictor_job_dev` (ID: `819238473316716`) executado com sucesso completo (`TERMINATED SUCCESS` no Run `442745640587439`).
  - Todas as 6 tarefas executadas com sucesso:
    1. `setup_environment`: DDL idempotente no Unity Catalog (`pitch_predictor.baseball_ml_dev`).
    2. `data_ingestion`: Ingestão incremental de lances Statcast via PyBaseball.
    3. `bronze_declarative_pipeline`: Pipeline Lakeflow Spark Declarative (`raw_pitches` $\rightarrow$ `silver_pitch_features`).
    4. `video_biomechanics_pipeline`: Ingestão incremental de clipes Savant, extração de 33 marcos 3D MediaPipe Pose e materialização na tabela Delta `biomechanical_features`.
    5. `automl_training`: Treinamento distribuído com split temporal por `game_date` e rastreamento MLflow.
    6. `model_inference`: Validação de inferência em lote e scoring com `@prod`.
- **Aplicação em Nuvem Operacional (Databricks Apps)**:
  - App `pitch-predictor-app-dev` em estado **`RUNNING` / `ACTIVE`**.
  - URL Corporativa Pública com Single Sign-On: **`https://pitch-predictor-app-dev-7474645845452998.aws.databricksapps.com`**.
  - Todas as 5 páginas Streamlit ativas em nuvem: Apresentação, Minigame Arcade, Simulador Tático, Scouting Report e Laboratório de Visão/Tipping.
  - Ponto de soltura (Release Point) calibrado com imagens broadcast HD de Paul Skenes no montinho, com retículo neon e HUD em todas as categorias.
- **Unity Catalog Dedicado**: Catálogo `pitch_predictor` ativo em Dev (`pitch_predictor.baseball_ml_dev`):
  - `raw_pitches`: Streaming Table Delta Live Tables (Bronze).
  - `silver_pitch_features`: Materialized View Delta Live Tables (Silver, zero-leakage).
  - `biomechanical_features`: Managed Delta Table com marcos 3D e ângulos cinemáticos (Gold/Biomechanical).
- **Modelo Multimodal Servido em Tempo Real (Databricks Model Serving Serverless)**:
  - Modelo registrado no Unity Catalog: `pitch_predictor.baseball_ml_dev.multimodal_pitch_predictor` (Versão 2, Alias `@prod`).
  - Endpoint ativo e operacional: `dev_rhian_souza_pitch_predictor_endpoint_dev` em estado `READY` / `DEPLOYMENT_READY` com escala para zero e 100% de tráfego.
  - Empacotamento MLflow Custom PyFunc (`MultimodalServingPyfuncModel` em `src/pitch_predictor/serving_model.py`):
    - Assinatura com colunas biomecânicas opcionais (`required=False`), garantindo que tanto chamadas puramente táticas quanto chamadas com cinemática corporal completa sejam atendidas com inferência em tempo real.
    - Saída estruturada retornando `prediction`, `prob_fastball`, `prob_breaking_ball`, `prob_offspeed`, `statcast_prediction`, `vision_prediction`, `alignment_status` e `summary_text`.
- **Módulo de Visão Computacional Biomecânica (`pitch_predictor.vision`)**:
  1. `video_ingestion.py`: Ingestão incremental de clipes de vídeo do Baseball Savant, upload híbrido (SDK/POSIX) para Unity Catalog Volume e manutenção de manifesto.
  2. `pose_extractor.py`: Extração de 33 marcos 3D (MediaPipe Pose), ângulos de braço (*arm slot*), altura de soltura, flexão de cotovelo, inclinação de tronco, extensão de passada e detecção delimitada do frame de soltura (*pitch tipping*).
  3. `tipping_model.py`: Classificador Random Forest de Pitch Tipping treinado em features biomecânicas com explicabilidade via SHAP TreeExplainer, gerando diagnósticos legíveis sobre quais movimentos corporais denunciam o tipo de arremesso.
  4. `overlay.py`: Sobreposição gráfica do esqueleto anatômico em neon Statcast sobre o corpo do arremessador no frame exato de soltura.
- **Módulo de Fusão Multimodal (`pitch_predictor.multimodal`)**:
  - `multimodal.py`: Fusão determinística e livre de vazamento (`(game_pk, at_bat_number, pitch_number)`), unindo features contextuais do Statcast (contagem, corredores, matchup, priors) e features biomecânicas (arm slot, flexão de cotovelo, altura).
  - `MultimodalPitchClassifier`: Classificador híbrido integrando pipelines especializados com estudo de ablação comparativo (*Ablation Study*) calculando o ganho de informação (*Lift*), diagnóstico de alinhamento tático vs tipping override e serialização `joblib`.
- **Camada de Apresentação (Streamlit)**: Arquitetura multipágina moderna com 5 módulos operacionais:
  1. `pages/00_home_apresentacao.py`: Apresentação executiva e amigável para leigos com 4 perfis de arremessador e 3 momentos táticos.
  2. `pages/01_live_predictor.py`: Simulador tático lance a lance com controle de contagem, corredores em base e inferência em tempo real.
  3. `pages/02_scout_report.py`: Scouting report completo com análise de tendências na matriz 4x3 de 12 contagens e plano de jogo.
  4. `pages/03_minigame.py`: Minigame retrô arcade (8-bit / 16-bit) com máquina de estados, sistema de atributos RPG (100 pontos), sistema de vidas, multiplicador de combo e áudio sintetizado Chiptune.
  5. `pages/04_pitch_vision_lab.py`: Laboratório biomecânico com player de vídeo, anotação esquelética no frame de soltura, comparador tri-modal conectado em tempo real ao Databricks Model Serving (com badge de latência em milissegundos e fallback local) e estudo de ablação interativo.
- **Camada de Inferência Resiliente**: `PitchServingClient` com suporte a:
  - `predict()`: Inferência retrocompatível para consumo dos módulos táticos e minigame.
  - `predict_multimodal()` e `predict_multimodal_batch()`: Inferência multimodal completa retornando probabilidades tri-modais, status de alinhamento e explicação textual gerada pelo modelo.
  - Modo Mock Offline inteligente para continuidade da experiência local caso as credenciais da nuvem não estejam disponíveis.

---

## 2. Invariantes de Arquitetura Invioláveis

Todo agente atuando neste projeto **DEVE SEGUIR ESTRITAMENTE** as 5 invariantes abaixo:

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

### Invariante 5: Isolamento de Escopo de UI & Estabilidade Visual (No-Unintended-Regressions)
- **Escopo Estrito de Telas**: Quando o usuário solicitar alterações de UI ou novos recursos direcionados a uma página específica (ex: minigame), é PROIBIDO introduzir novos seletores, cards ou reformulações visuais em páginas maduras e estáveis sem instrução expressa.
- **Não Aninhamento Excessivo no Streamlit**: `st.columns()` NUNCA deve ser chamado dentro de outra coluna (`st.columns` suporta no máximo 1 nível de profundidade). Aninhamentos maiores causam `StreamlitAPIException` fatal.
- **Validação de Opções em Widgets**: Qualquer chave de `st.session_state` vinculada a `st.selectbox`, `st.radio` ou `st.multiselect` deve ser validada contra a lista de opções ativas antes da renderização para prevenir `ValueError: ... is not in iterable`.

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
│   ├── 03_model_inference.py     # Inferência em lote e validação de métricas MLflow
│   └── 04_video_biomechanics_pipeline.py # Tarefa Lakeflow Job para ingestão incremental de vídeo e pose
├── pages/
│   ├── 00_home_apresentacao.py   # Apresentação executiva com perfis oficiais e predição tática
│   ├── 02_scout_report.py        # Scouting report detalhado com matriz 4x3 de 12 contagens
│   ├── 03_minigame.py            # Minigame retrô arcade (Você vs. IA no Montinho)
│   └── 04_pitch_vision_lab.py    # Laboratório de Visão Computacional, Tipping & Comparador Tri-Modal
├── resources/
│   ├── pitch_predictor_app.yml   # Definição do Databricks App
│   ├── pitch_predictor_endpoint.yml # Definição do Endpoint de Model Serving Serverless
│   ├── pitch_predictor_job.yml   # Definição do Job com 3 tarefas orquestradas
│   └── pitch_predictor_pipeline.pipeline.yml # Definição do Pipeline Declarativo Lakeflow
├── scripts/
│   ├── download_pitch_videos.py  # Download incremental e manifesto de clipes Savant
│   ├── extract_pitch_poses.py    # Extração de landmarks MediaPipe e métricas cinemáticas
│   ├── generate_all_15_release_frames.py # Geração e calibração de snapshots em neon Statcast
│   ├── ingest_multiple_pitchers.py # Ingestão incremental dos 5 arremessadores de elite
│   ├── register_multimodal_model.py # Registro do PyFunc Multimodal no Unity Catalog com @prod
│   ├── setup_unity_catalog.py    # Utilitário CLI para inicialização do Unity Catalog
│   ├── setup_unity_catalog.sql   # DDL SQL idempotente com tokens de catálogo/esquema
│   ├── train_multimodal_model.py # Treinamento do modelo híbrido e estudo de ablação (Lift)
│   ├── train_tipping_model.py    # Treinamento do classificador de Pitch Tipping com SHAP
│   ├── verify_browser_dom.py     # Script de inspeção de DOM das páginas Streamlit
│   ├── verify_minigame_browser.py# Teste E2E headless do minigame arcade via Playwright
│   └── verify_ui_server.py       # Probe de saúde HTTP da UI
├── src/pitch_predictor/
│   ├── config.py                 # Resolução de variáveis e cliente WorkspaceClient
│   ├── fallback_engine.py        # Motor unificado de inferência offline & fallback tático/multimodal
│   ├── features.py               # Colunas Silver, mapeamentos e transformações zero-leakage
│   ├── ingestion.py              # Download de dados Statcast e sanitização inicial
│   ├── minigame_engine.py        # Motor de jogo puro (regras, máquina de estados, resolução de duelos)
│   ├── multimodal.py             # Fusão multimodal determinística, Ablation Study & Lift
│   ├── pipelines/bronze_pitches.py # Pipeline Lakeflow DLT para camada Bronze
│   ├── scenarios.py              # Dicionários de cenários, priors e registro canônico MLB_PITCHERS
│   ├── scout_analytics.py        # Agregações analíticas e cálculo da matriz de contagens
│   ├── serving_client.py         # Cliente de inferência (Databricks Serving com retry e backoff)
│   ├── serving_model.py          # MLflow PyFunc Multimodal com fallback de imputação e diagnóstico
│   ├── training.py               # Orquestração do AutoML e rastreamento de experimentos MLflow
│   ├── ui_components.py          # Componentes visuais SVG (Campo de Beisebol e Strike Zone)
│   ├── utils.py                  # Logging formatado e sessão Spark local
│   └── vision/                   # Subpacote de Visão Computacional & Biomecânica
│       ├── __init__.py           # Exports unificados e detecção graciosa de deps
│       ├── incremental_pipeline.py # Orquestrador de ingestão incremental no Databricks
│       ├── overlay.py            # Sobreposição esquelética neon Statcast
│       ├── pose_extractor.py     # Extração de 33 landmarks 3D MediaPipe
│       ├── tipping_model.py      # PitchTippingClassifier & SHAP TreeExplainer
│       └── video_ingestion.py    # Ingestão de vídeos Savant & upload UC Volume
└── tests/                        # 202 testes automatizados (pytest)
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