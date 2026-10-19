---
name: streamlit
description: Comprehensive guide and best practices for developing high-performance, responsive Streamlit applications. Covers reactive architecture, session state management, widget callback patterns, custom CSS/HTML styling, layout organization, AppTest headless testing framework, and external API/Databricks model serving integration with cold-start resilience.
license: Apache-2.0
allowed-tools: Read Write Edit Bash
compatibility: Requires Python 3.10+ and streamlit>=1.30.0.
metadata:
  version: "1.0"
  last-reviewed: "2026-10-17"
  skill-author: Antigravity Agentic Team
---

# Streamlit Development & Best Practices

## 1. Overview
Streamlit is the standard Python framework for building interactive data science, machine learning, and tactical simulation web apps. This skill guides the architecture, state handling, UI styling, and automated testing of production-grade Streamlit applications.

---

## 2. Core Architectural Patterns

### 2.1. Reactive Execution & Session State
Streamlit executes top-to-bottom on every user interaction. To maintain stability across reruns:
- **Always initialize state variables in `st.session_state`**:
  ```python
  if "balls" not in st.session_state:
      st.session_state.balls = 0
  if "strikes" not in st.session_state:
      st.session_state.strikes = 0
  ```
- **Use Callbacks for State Mutations**: Rather than modifying widgets directly after rendering, attach `on_change` callbacks:
  ```python
  def on_scenario_selected():
      scenario = SCENARIOS[st.session_state.selected_scenario]
      st.session_state.balls = scenario["balls"]
      st.session_state.strikes = scenario["strikes"]

  st.selectbox("Cenário", options=list(SCENARIOS.keys()), key="selected_scenario", on_change=on_scenario_selected)
  ```
- **Avoid Key-Widget Overwrite Conflicts**: When programmatic state updates are required, update `st.session_state[key]` inside callbacks before the widget re-renders.

### 2.2. Page Layout & Information Hierarchy
- **Header & Metrics Row**:
  Use `st.set_page_config(page_title="...", layout="wide", initial_sidebar_state="expanded")`.
- **Multi-Column Grids**:
  Use `col1, col2, col3 = st.columns([1.2, 1.2, 1.0], gap="medium")` to balance density.
- **Visual Containers & Cards**:
  Group controls with `with st.container(border=True):` to create distinct visual cards.

---

## 3. Custom Styling & Visual Customization

### 3.1. Injecting Scoped CSS
Streamlit allows custom CSS via `st.markdown("<style>...</style>", unsafe_allow_html=True)`.
Key guidelines:
- Use consistent dark/tactic themes with high-contrast text.
- Target elements via semantic container classes or custom HTML wrappers:
  ```html
  <div class="tactical-card fastball">
      <div class="card-badge">PREDIÇÃO TÁTICA</div>
      <div class="pitch-name">⚡ Fastball</div>
  </div>
  ```
- Define CSS custom properties (variables) for palette consistency:
  ```css
  :root {
      --bg-dark: #0f172a;
      --card-bg: #1e293b;
      --fastball-red: #ef4444;
      --breaking-blue: #3b82f6;
      --offspeed-green: #10b981;
      --accent-gold: #f59e0b;
  }
  ```

### 3.2. Embedding Inline SVG Graphics
For custom diagrams (e.g., Baseball Diamond, Strike Zone):
- Render SVGs directly via `st.markdown(svg_string, unsafe_allow_html=True)`.
- Use `viewBox="0 0 W H"` for full vector responsiveness.
- Bind fill/stroke colors dynamically based on `st.session_state`.

---

## 4. Serving & Asynchronous Operations

### 4.1. Caching Resources
- Cache persistent connections (e.g. Databricks SDK `WorkspaceClient` or HTTP sessions):
  ```python
  @st.cache_resource(show_spinner=False)
  def get_serving_client():
      return PitchServingClient()
  ```

### 4.2. Cold-Start & Scale-to-Zero Handling
Serverless model serving endpoints may take 15–35 seconds to warm up from zero replicas.
- Always wrap model calls in a descriptive spinner:
  ```python
  with st.spinner("⏳ Conectando ao Databricks Serverless Model Serving... (Acordando contêiner)"):
      prediction = client.predict(features)
  ```
- Catch timeouts and HTTP 503 errors gracefully, offering a retry button instead of a crash trace.

---

## 5. Automated Headless Testing (`AppTest`)

Streamlit provides a native testing framework in `streamlit.testing.v1.AppTest`:
- Simulates user interactions and inspects widgets without opening a physical browser.
- Integrates seamlessly with `pytest`.

```python
from streamlit.testing.v1 import AppTest

def test_app_loads():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) > 0

def test_scenario_selection():
    at = AppTest.from_file("app.py").run()
    # Find selectbox and choose scenario
    at.selectbox[0].select("Contagem Cheia").run()
    assert not at.exception
```
