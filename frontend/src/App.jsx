import { useEffect, useMemo, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

const DEFAULT_SUGGESTIONS = [
  "Summarize all books.",
  "Who killed Snape?",
  "Compare the book and movie adaptations.",
  "List key moments from The Two Towers.",
];

const getInitialSessionId = () => {
  if (typeof window === "undefined") return "local-user";
  const stored = window.localStorage.getItem("bookMovieChatSessionId");
  if (stored) return stored;
  const newId = crypto.randomUUID();
  window.localStorage.setItem("bookMovieChatSessionId", newId);
  return newId;
};

const MessageBubble = ({ message }) => {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} w-full`}>
      <div
        className={`rounded-2xl px-4 py-3 max-w-xl shadow-md ${
          isUser
            ? "bg-primary text-white"
            : "bg-white/5 text-gray-100 border border-white/10"
        }`}
      >
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
        {isAssistant && message.corrected_query && (
          <p className="mt-2 text-xs text-accent">
            Interpreted your question as: <span className="italic">{message.corrected_query}</span>
          </p>
        )}
        {isAssistant && message.sources?.length > 0 && (
          <div className="mt-3 border-t border-white/10 pt-2">
            <p className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
              Sources
            </p>
            <ul className="mt-1 space-y-1 text-xs text-gray-300">
              {message.sources.map((source, idx) => (
                <li key={`${source.source}-${source.page}-${idx}`} className="flex gap-2">
                  <span className="text-primary-light">•</span>
                  <span>
                    {source.source} – page {source.page ?? "?"}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  const [sessionId] = useState(getInitialSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [suggestions, setSuggestions] = useState(DEFAULT_SUGGESTIONS);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/history?session_id=${sessionId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch history.");
        }
        const data = await response.json();
        setMessages(data.history ?? []);
      } catch (err) {
        console.error(err);
        setError("Could not load chat history. You can still start a new conversation.");
      }
    };

    loadHistory();
  }, [sessionId]);

  useEffect(() => {
    const loadSuggestions = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/suggest?session_id=${sessionId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch suggestions.");
        }
        const data = await response.json();
        if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
          setSuggestions(data.suggestions);
        }
      } catch (err) {
        console.warn("Suggestion fetch failed:", err);
      }
    };

    loadSuggestions();
  }, [sessionId]);

  const recentHistory = useMemo(
    () => messages.slice(-10).reverse(),
    [messages],
  );

  const sendMessage = async (overrideMessage) => {
    const rawMessage = overrideMessage ?? input;
    const trimmed = rawMessage.trim();
    if (!trimmed || loading) return;

    setInput("");
    setError("");
    setLoading(true);

    const userMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, query: trimmed }),
      });

      if (!response.ok) {
        throw new Error("Chat request failed.");
      }

      const data = await response.json();
      const assistantMessage = {
        role: "assistant",
        content: data.response ?? "I couldn't generate a response this time.",
        sources: data.sources ?? [],
        corrected_query: data.corrected_query ?? "",
        query_type: data.query_type ?? "",
      };

      setMessages((prev) => [...prev, assistantMessage]);
      if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
        setSuggestions(data.suggestions);
      }
    } catch (err) {
      console.error(err);
      setError("Something went wrong reaching the assistant. Please try again.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I had trouble connecting to the server. Please try again shortly.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    void sendMessage();
  };

  return (
    <div className="bg-gradient-to-br from-surface via-[#11111d] to-[#1b1b2d] min-h-screen text-gray-100">
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col lg:flex-row gap-6">
        <section className="flex-1 bg-white/5 border border-white/10 rounded-3xl shadow-xl flex flex-col overflow-hidden">
          <header className="px-6 py-5 border-b border-white/10 bg-white/5 backdrop-blur">
            <h1 className="text-2xl font-semibold text-white">BookMovieChat</h1>
            <p className="text-sm text-gray-300 mt-1">
              Chat with your favorite book and movie PDFs. Ask questions, get summaries, and compare adaptations.
            </p>
          </header>

          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4 scroll-smooth">
            {messages.length === 0 && !loading ? (
              <div className="text-center text-gray-400 text-sm">
                <p>Upload a PDF in the backend and start the conversation with one of the suggestions below.</p>
              </div>
            ) : (
              messages.map((message, index) => <MessageBubble key={index} message={message} />)
            )}

            {loading && (
              <div className="flex items-center gap-3 text-sm text-gray-300">
                <span className="relative flex h-3 w-3">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary-light opacity-75" />
                  <span className="relative inline-flex h-3 w-3 rounded-full bg-primary" />
                </span>
                Thinking...
              </div>
            )}
          </div>

          {error && (
            <div className="px-6 pb-2">
              <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2">
                {error}
              </p>
            </div>
          )}

          <form
            onSubmit={handleSubmit}
            className="px-6 pt-4 pb-6 border-t border-white/10 bg-black/30 backdrop-blur"
          >
            <div className="flex gap-3 items-end">
              <textarea
                className="flex-1 resize-none rounded-2xl bg-white/10 border border-white/10 p-4 text-sm shadow-inner focus:outline-none focus:ring-2 focus:ring-primary-light focus:border-transparent transition"
                rows={3}
                placeholder="Ask about a scene, summarize a chapter, or compare book and film..."
                value={input}
                onChange={(event) => setInput(event.target.value)}
                disabled={loading}
              />
              <button
                type="submit"
                className="h-12 px-6 rounded-2xl bg-primary hover:bg-primary-dark transition text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={loading || input.trim().length === 0}
              >
                Send
              </button>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="text-xs px-4 py-2 rounded-full bg-white/5 border border-white/10 hover:bg-primary-light/10 transition text-gray-300"
                  onClick={() => void sendMessage(suggestion)}
                  disabled={loading}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </form>
        </section>

        <aside className="lg:w-80 bg-white/5 border border-white/10 rounded-3xl shadow-xl p-6 space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Recent History</h2>
            <p className="text-xs text-gray-400">
              Last 10 messages in this session.
            </p>
          </div>
          <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
            {recentHistory.length === 0 ? (
              <p className="text-sm text-gray-400">
                No chat history yet. Start the conversation!
              </p>
            ) : (
              recentHistory.map((item, idx) => (
                <div
                  key={`${item.role}-${idx}`}
                  className="bg-black/30 border border-white/10 rounded-2xl px-3 py-2"
                >
                  <p className="text-xs uppercase tracking-wide text-gray-400 mb-1">
                    {item.role}
                  </p>
                  <p className="text-sm text-gray-100">{item.content}</p>
                </div>
              ))
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
