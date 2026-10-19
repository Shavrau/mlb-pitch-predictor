---
name: ui-browser-testing
description: Methodologies and automated protocols for testing and verifying Streamlit and web UI applications in headless environments and real browsers. Covers Streamlit AppTest native automation, Playwright end-to-end browser testing, DOM and visual inspection, network interception for Databricks model endpoints, and health probe validation.
license: Apache-2.0
allowed-tools: Read Write Edit Bash
metadata:
  version: "1.0"
  last-reviewed: "2026-10-17"
  skill-author: Antigravity Agentic Team
---

# UI & Browser Verification: Streamlit & Web Applications

## 1. Multi-Layer Testing Architecture

Para aplicações Streamlit conectadas a serviços em nuvem (Databricks Serverless), a verificação deve ocorrer em três camadas complementares:

```
+-------------------------------------------------------------+
| Camada 3: Real Browser E2E (Playwright / Headless Chromium) |
| - Validação de renderização gráfica (SVG Diamond, CSS Grid) |
| - Interações de clique, scroll e carregamento de fontes     |
+-------------------------------------------------------------+
                              ▲
+-------------------------------------------------------------+
| Camada 2: Streamlit AppTest (Native Pytest Automation)     |
| - Simulação headless de widgets (sliders, botões, selects)  |
| - Inspeção de st.session_state e asserção de ausência de    |
|   erros/exceções                                            |
+-------------------------------------------------------------+
                              ▲
+-------------------------------------------------------------+
| Camada 1: Contrato Unitário & Mock de Endpoint              |
| - Validação das 24 features e tipos exatos                  |
| - Simulação de cold-start, timeouts e respostas do modelo   |
+-------------------------------------------------------------+
```

---

## 2. Camada 1 & 2: Automação Headless com `AppTest`

O `AppTest` executa o script Streamlit em um ambiente Python simulado sem a sobrecarga de iniciar um navegador físico:

```python
from streamlit.testing.v1 import AppTest

def test_app_initial_render():
    at = AppTest.from_file("app.py", default_timeout=20).run()
    assert not at.exception, f"App levantou exceção: {at.exception}"
    assert "Pitch Predictor" in at.title[0].value

def test_scenario_switch_updates_count():
    at = AppTest.from_file("app.py").run()
    # Encontra o seletor de cenário
    at.selectbox[0].select("Contagem Cheia").run()
    assert not at.exception
```

---

## 3. Camada 3: Verificação Ponta a Ponta no Navegador

### 3.1. Subida do Servidor em Background
Executar o Streamlit em modo headless desativando telemetria e navegador automático:
```bash
uv run streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
```

### 3.2. Probes de Saúde HTTP & WebSocket
Verificar se o servidor Streamlit está respondendo:
- **Health Check**: `GET http://localhost:8501/_stcore/health` (deve retornar HTTP 200 com body `"ok"`).
- **Host Status**: `GET http://localhost:8501/` (deve conter `<title>` e tags do Streamlit bundle).

### 3.3. Testes End-to-End com Playwright (ou Scripts Headless)
Automação com Playwright em modo headless para validar que:
1. O SVG do Diamante de Bases é renderizado com elementos `<polygon>` e `<rect>`.
2. A grade Statcast (zonas 1 a 14) é renderizada e responde a cliques.
3. O botão de predição dispara a consulta e atualiza o card de predição com a cor correta.
4. O spinner de cold start é exibido durante a espera e ocultado após a resposta.
