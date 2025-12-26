"""
Professional Trading Dashboard Theme
=====================================
Custom CSS for a polished, professional trading platform look.
"""

DARK_THEME_CSS = """
<style>
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --bg-card: #1c2128;
    --border-color: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-yellow: #d29922;
    --accent-purple: #a371f7;
}

.stApp { background: linear-gradient(180deg, #1a1f29 0%, #0d1117 100%); }
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
}

.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(88, 166, 255, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.main-header h1 { color: #fff; font-weight: 700; font-size: 1.8rem; margin: 0; }
.main-header .subtitle { color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem; }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover { border-color: var(--accent-blue); box-shadow: 0 4px 12px rgba(88, 166, 255, 0.15); }
.metric-value { font-size: 1.75rem; font-weight: 700; color: var(--text-primary); margin: 0.5rem 0; }
.metric-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; }
.metric-positive { color: var(--accent-green) !important; }
.metric-negative { color: var(--accent-red) !important; }
.metric-neutral { color: var(--accent-yellow) !important; }

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}
.status-running { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }
.status-stopped { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 81, 73, 0.3); }
.status-warning { background: rgba(210, 153, 34, 0.15); color: var(--accent-yellow); border: 1px solid rgba(210, 153, 34, 0.3); }

.stTabs [data-baseweb="tab-list"] { background: var(--bg-secondary); border-radius: 8px; padding: 0.25rem; gap: 0.25rem; }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--text-secondary); border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500; }
.stTabs [aria-selected="true"] { background: var(--bg-tertiary); color: var(--accent-blue); }

.stButton > button { border-radius: 8px; font-weight: 600; padding: 0.5rem 1.25rem; transition: all 0.2s ease; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }

.section-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
.section-title { color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color); }

.stDataFrame { border-radius: 8px; overflow: hidden; }
.stDataFrame [data-testid="stDataFrame"] { background: var(--bg-card); border: 1px solid var(--border-color); }

.stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 6px; color: var(--text-primary); }
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: var(--accent-blue); box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2); }

.stAlert { border-radius: 8px; border: none; }
.stSuccess { background: rgba(63, 185, 80, 0.1); border-left: 4px solid var(--accent-green); }
.stWarning { background: rgba(210, 153, 34, 0.1); border-left: 4px solid var(--accent-yellow); }
.stError { background: rgba(248, 81, 73, 0.1); border-left: 4px solid var(--accent-red); }
.stInfo { background: rgba(88, 166, 255, 0.1); border-left: 4px solid var(--accent-blue); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

.long-position, .profit { color: var(--accent-green); }
.short-position, .loss { color: var(--accent-red); }

@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
.live-indicator { animation: pulse 2s infinite; }

.grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
</style>
"""


def get_full_css() -> str:
    """Return the complete CSS styling."""
    return DARK_THEME_CSS


def metric_card(label: str, value: str, status: str = "neutral") -> str:
    """Generate HTML for a styled metric card."""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value metric-{status}">{value}</div></div>'


def status_badge(text: str, status: str = "running") -> str:
    """Generate HTML for a status badge."""
    return f'<span class="status-badge status-{status}">{text}</span>'


def header_html(title: str, subtitle: str = "") -> str:
    """Generate HTML for the main header."""
    return f'<div class="main-header"><h1>🤖 {title}</h1><div class="subtitle">{subtitle}</div></div>'


def section_card(title: str, content: str) -> str:
    """Generate HTML for a section card."""
    return f'<div class="section-card"><div class="section-title">{title}</div>{content}</div>'

