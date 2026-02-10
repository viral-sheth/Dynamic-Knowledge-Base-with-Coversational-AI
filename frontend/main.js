const { useState, useEffect, useRef, useMemo } = React;

const storageKey = "healthcare-rag-api-base";
const themeKey = "healthcare-rag-theme";
const promptSuggestions = [
  "What is the timely filing deadline for UnitedHealthcare Community Plan in Florida?",
  "Which services require prior authorization for UnitedHealthcare Community Plan Arizona?",
  "What documentation is required for claims submission in Arizona?",
  "How do I submit an appeal for a denied claim with UnitedHealthcare?",
];

function ChatApp() {
  const [apiBase, setApiBase] = useState(
    localStorage.getItem(storageKey) || "http://localhost:8000"
  );
  const [theme, setTheme] = useState(localStorage.getItem(themeKey) || "light");
  const [input, setInput] = useState(
    ""
  );
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "bot",
      text: "Hello! I can help you find information about healthcare policies. Ask me about claims, appeals, prior authorization, or timely filing requirements.",
      sources: [],
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatRef = useRef(null);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.classList.toggle("light", theme === "light");
    localStorage.setItem(themeKey, theme);
  }, [theme]);

  useEffect(() => {
    const el = chatRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    const userMsg = { id: crypto.randomUUID(), role: "user", text: question };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const base = apiBase.replace(/\/+$/, "");
      localStorage.setItem(storageKey, base);

      const res = await fetch(`${base}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`API error ${res.status}: ${detail}`);
      }

      const data = await res.json();
      const botMsg = {
        id: crypto.randomUUID(),
        role: "bot",
        text: data.answer || "No answer returned.",
        sources: data.sources || [],
      };
      setMessages((m) => [...m, botMsg]);
    } catch (err) {
      const botMsg = {
        id: crypto.randomUUID(),
        role: "bot",
        text: `Failed to fetch: ${err.message}`,
        sources: [],
      };
      setMessages((m) => [...m, botMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([
      {
        id: "welcome",
        role: "bot",
        text: "Chat cleared. Ask anything about healthcare rules.",
        sources: [],
      },
    ]);
  };

  const copyText = (text) => {
    navigator.clipboard?.writeText(text).catch(() => {});
  };

  const typingIndicator = useMemo(
    () =>
      isLoading ? (
        <div className="msg bot">
          <div className="typing">
            <span>Bot is typing</span>
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        </div>
      ) : null,
    [isLoading]
  );

  return (
    <div className="app">
      <div className="topbar">
        <div className="title">
          <h1>Healthcare Policy Assistant</h1>
          <p>Conversational search with cited policy sources</p>
        </div>
        <div className="controls">
          <button
            className="btn"
            title="Settings"
            onClick={() => setShowSettings((v) => !v)}
          >
            ⚙️
          </button>
          <button
            className="btn"
            title="Toggle theme"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
          <button className="btn" title="Clear chat" onClick={handleClear}>
            🗑️
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="config">
          <label>API base</label>
          <input
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="http://localhost:8000"
          />
          <button className="btn primary" onClick={() => localStorage.setItem(storageKey, apiBase)}>
            Save
          </button>
        </div>
      )}

      <div className="chat-shell">
        <div className="chat-window" ref={chatRef}>
          {messages.map((m) => (
            <div key={m.id} className={`msg ${m.role}`}>
              {m.role === "bot" && (
                <button className="copy-btn" onClick={() => copyText(m.text)}>
                  Copy
                </button>
              )}
              <div className="meta">{m.role === "user" ? "You" : "Assistant"}</div>
              <div className="msg-content">{m.text}</div>
              {m.sources && m.sources.length > 0 && (
                <div className="sources">
                  {m.sources.map((s, idx) => (
                    <div key={idx} className="source-pill">
                      <strong>{s.title || "Policy rule"}</strong>
                      {(s.filename || s.page_number || s.rule_type || s.payer_name) && (
                        <span style={{ fontSize: "12px", color: "var(--muted)" }}>
                          {s.filename ? `File: ${s.filename}` : ""}
                          {typeof s.page_number === "number" ? ` · Page ${s.page_number}` : ""}
                          {s.rule_type ? ` · Type: ${s.rule_type}` : ""}
                          {s.payer_name ? ` · Payer: ${s.payer_name}` : ""}
                        </span>
                      )}
                      {s.url ? (
                        <a href={s.url} target="_blank" rel="noreferrer">
                          Open source
                        </a>
                      ) : (
                        <span style={{ color: "var(--muted)", fontSize: "12px" }}>
                          No URL provided
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {typingIndicator}
        </div>

        <div className="composer">
          <div className="input-row">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about prior auth, appeals, claims submission, or payer-specific rules..."
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <button className="btn primary" onClick={handleSend} disabled={isLoading}>
              {isLoading ? "Sending..." : "Send"}
            </button>
          </div>
          <div className="footer-row">
            <div className="chips">
              {promptSuggestions.map((p, idx) => (
                <button key={idx} className="chip" onClick={() => setInput(p)}>
                  {p}
                </button>
              ))}
            </div>
            <div className="typing">{isLoading ? "Bot is typing..." : "Ready"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<ChatApp />);
