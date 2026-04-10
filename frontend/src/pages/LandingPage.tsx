import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Bot, Zap, GitBranch, Terminal, Globe, Brain,
  ArrowRight, Github, Star, ChevronRight, Code2,
  FileCode, Cpu, Shield, Layers, CheckCircle2, Sun, Moon
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

const DEMO_MESSAGES = [
  { role: 'user', text: 'Build a REST API for a todo app with FastAPI' },
  { role: 'agent', text: 'Planning the project structure...', phase: 'Planning' },
  { role: 'agent', text: 'Creating main.py, models.py, routes.py...', phase: 'Executing' },
  { role: 'agent', text: 'Writing pytest tests for all endpoints...', phase: 'Verifying' },
  { role: 'agent', text: 'All 12 tests passed. API ready! ✅', phase: 'Complete' },
];

const FEATURES = [
  {
    icon: <Brain size={20} />,
    title: 'Autonomous Agent',
    desc: 'Plans, executes, and verifies tasks end-to-end without manual intervention.',
    color: 'text-[#00ff99]',
    bg: 'bg-[#00ff99]/10',
  },
  {
    icon: <Terminal size={20} />,
    title: 'Real Shell Access',
    desc: 'Runs terminal commands, installs packages, and manages files directly.',
    color: 'text-blue-400',
    bg: 'bg-blue-400/10',
  },
  {
    icon: <GitBranch size={20} />,
    title: 'Git Integration',
    desc: 'Auto-commits changes, creates branches, and opens pull requests on GitHub.',
    color: 'text-orange-400',
    bg: 'bg-orange-400/10',
  },
  {
    icon: <Globe size={20} />,
    title: 'Web Browsing',
    desc: 'Searches the web and reads documentation to solve real-world problems.',
    color: 'text-purple-400',
    bg: 'bg-purple-400/10',
  },
  {
    icon: <Cpu size={20} />,
    title: 'Multi-Model',
    desc: 'Supports GPT-4o, Claude 3.5, Gemini 1.5 Pro — switch anytime.',
    color: 'text-yellow-400',
    bg: 'bg-yellow-400/10',
  },
  {
    icon: <Shield size={20} />,
    title: 'Self-Correcting',
    desc: 'Detects errors, retries with a different approach, never gives up.',
    color: 'text-red-400',
    bg: 'bg-red-400/10',
  },
];

const EXAMPLE_PROMPTS = [
  'Build a FastAPI CRUD app with SQLite',
  'Fix all bugs in my Python script',
  'Add authentication to my Express server',
  'Write unit tests for this React component',
  'Scrape product prices from Amazon',
  'Deploy my app to a Linux server',
];

const MODELS = [
  { name: 'GPT-4o', provider: 'OpenAI', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
  { name: 'Claude 3.5', provider: 'Anthropic', color: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
  { name: 'Gemini 1.5', provider: 'Google', color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' },
];

function TypewriterText({ text, speed = 30 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    setDisplayed('');
    let i = 0;
    const id = setInterval(() => {
      setDisplayed(text.slice(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(id);
    }, speed);
    return () => clearInterval(id);
  }, [text, speed]);
  return <span>{displayed}<span className="animate-pulse">|</span></span>;
}

export default function LandingPage() {
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();
  const [demoStep, setDemoStep] = useState(0);
  const [promptIdx, setPromptIdx] = useState(0);
  const demoRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    demoRef.current = setInterval(() => {
      setDemoStep(s => (s + 1) % DEMO_MESSAGES.length);
    }, 2500);
    return () => { if (demoRef.current) clearInterval(demoRef.current); };
  }, []);

  useEffect(() => {
    const id = setInterval(() => setPromptIdx(p => (p + 1) % EXAMPLE_PROMPTS.length), 3000);
    return () => clearInterval(id);
  }, []);

  const bg = isDark ? 'bg-[#0a0a0a]' : 'bg-[#f8f8f8]';
  const text = isDark ? 'text-white' : 'text-[#0f0f0f]';
  const textMuted = isDark ? 'text-[#737373]' : 'text-[#525252]';
  const cardBg = isDark ? 'bg-[#111111] border-[#1a1a1a]' : 'bg-white border-[#e5e5e5]';
  const navBg = isDark ? 'bg-[#0a0a0a]/80 border-[#1a1a1a]' : 'bg-white/80 border-[#e5e5e5]';

  return (
    <div className={`min-h-screen ${bg} ${text} font-sans overflow-x-hidden`}>
      {/* Nav */}
      <nav className={`fixed top-0 left-0 right-0 z-50 border-b ${navBg} backdrop-blur-md`}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-[#00ff99]/10 border border-[#00ff99]/20 flex items-center justify-center">
              <Bot className="text-[#00ff99]" size={18} />
            </div>
            <span className="font-bold text-base">Mini-Devin</span>
            <span className="hidden sm:block text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 bg-[#00ff99]/10 text-[#00ff99] rounded-full border border-[#00ff99]/20">
              Beta
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg ${textMuted} hover:text-[#00ff99] transition-colors`}
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <a
              href="https://github.com/naimprince010-ship-it/mini-devin"
              target="_blank"
              className={`hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${textMuted} hover:text-white transition-colors`}
            >
              <Github size={15} />
              GitHub
            </a>
            <button
              onClick={() => navigate('/app')}
              className="flex items-center gap-2 px-4 py-2 bg-[#00ff99] text-[#0f0f0f] text-sm font-bold rounded-xl hover:bg-[#00e589] transition-colors"
            >
              Launch App
              <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#00ff99]/5 border border-[#00ff99]/15 mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse" />
            <span className="text-[#00ff99] text-xs font-bold uppercase tracking-widest">Autonomous AI Engineer</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-6xl font-black tracking-tight leading-tight mb-6">
            Your AI that{' '}
            <span className="text-[#00ff99]">actually codes</span>
            <br />
            — not just talks
          </h1>

          <p className={`text-lg sm:text-xl ${textMuted} max-w-2xl mx-auto leading-relaxed mb-10`}>
            Mini-Devin is an autonomous software engineer agent that plans, writes code, runs tests,
            fixes bugs, and commits to GitHub — all on its own.
          </p>

          {/* Rotating prompt */}
          <div className={`inline-flex items-center gap-3 px-4 py-3 rounded-xl ${isDark ? 'bg-[#111111] border border-[#262626]' : 'bg-white border border-[#e5e5e5]'} mb-10 max-w-lg w-full mx-auto`}>
            <Terminal size={16} className="text-[#00ff99] flex-shrink-0" />
            <span className="text-sm font-mono text-[#a3a3a3] text-left">
              <TypewriterText key={promptIdx} text={EXAMPLE_PROMPTS[promptIdx]} speed={40} />
            </span>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate('/app')}
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-3.5 bg-[#00ff99] text-[#0f0f0f] font-bold text-base rounded-2xl hover:bg-[#00e589] hover:scale-105 transition-all shadow-lg shadow-[#00ff99]/20"
            >
              <Zap size={18} fill="currentColor" />
              Start Building Free
            </button>
            <a
              href="https://github.com/naimprince010-ship-it/mini-devin"
              target="_blank"
              className={`w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-3.5 rounded-2xl border ${isDark ? 'border-[#262626] hover:border-[#363636] text-[#a3a3a3] hover:text-white' : 'border-[#d4d4d4] hover:border-[#a3a3a3] text-[#525252] hover:text-[#0f0f0f]'} font-semibold text-base transition-all`}
            >
              <Star size={16} />
              Star on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* Live Demo Terminal */}
      <section className="py-16 px-6">
        <div className="max-w-3xl mx-auto">
          <div className={`rounded-2xl border overflow-hidden shadow-2xl ${isDark ? 'border-[#1a1a1a]' : 'border-[#e5e5e5]'}`}>
            {/* Terminal header */}
            <div className={`flex items-center gap-2 px-4 py-3 ${isDark ? 'bg-[#161616] border-b border-[#1a1a1a]' : 'bg-[#f0f0f0] border-b border-[#e5e5e5]'}`}>
              <div className="w-3 h-3 rounded-full bg-red-500/60" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
              <div className="w-3 h-3 rounded-full bg-green-500/60" />
              <span className={`ml-2 text-xs ${textMuted} font-mono`}>mini-devin — session</span>
              <div className="ml-auto flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse" />
                <span className="text-[10px] text-[#00ff99] font-bold uppercase tracking-wider">Live</span>
              </div>
            </div>

            {/* Messages */}
            <div className={`p-6 space-y-4 min-h-[200px] ${isDark ? 'bg-[#0d0d0d]' : 'bg-white'}`}>
              {DEMO_MESSAGES.slice(0, demoStep + 1).map((msg, i) => (
                <div key={i} className={`flex gap-3 items-start ${msg.role === 'user' ? 'justify-end' : ''}`}>
                  {msg.role === 'agent' && (
                    <div className="w-7 h-7 rounded-lg bg-[#00ff99]/10 border border-[#00ff99]/20 flex items-center justify-center flex-shrink-0">
                      <Bot size={14} className="text-[#00ff99]" />
                    </div>
                  )}
                  <div className={`max-w-md px-4 py-2.5 rounded-xl text-sm ${
                    msg.role === 'user'
                      ? 'bg-[#00ff99] text-[#0f0f0f] font-semibold rounded-tr-sm'
                      : isDark
                        ? 'bg-[#111111] border border-[#1a1a1a] text-[#c0c0c0] rounded-tl-sm'
                        : 'bg-[#f5f5f5] border border-[#e5e5e5] text-[#333] rounded-tl-sm'
                  }`}>
                    {msg.phase && (
                      <span className="text-[10px] font-bold uppercase tracking-widest text-[#00ff99]/70 block mb-1">
                        [{msg.phase}]
                      </span>
                    )}
                    {i === demoStep && msg.role === 'agent'
                      ? <TypewriterText text={msg.text} speed={25} />
                      : msg.text}
                  </div>
                  {msg.role === 'user' && (
                    <div className="w-7 h-7 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center flex-shrink-0 text-xs font-bold text-white bg-[#2a2a2a]">
                      U
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
          <p className={`text-center mt-4 text-xs ${textMuted}`}>Live demo — watch the agent work in real time</p>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-black tracking-tight mb-4">
              Everything a senior engineer does
            </h2>
            <p className={`text-lg ${textMuted} max-w-xl mx-auto`}>
              Not just code generation — real autonomous engineering.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f, i) => (
              <div key={i} className={`p-6 rounded-2xl border ${cardBg} hover:border-[#00ff99]/20 transition-all group`}>
                <div className={`w-10 h-10 rounded-xl ${f.bg} flex items-center justify-center mb-4 ${f.color} group-hover:scale-110 transition-transform`}>
                  {f.icon}
                </div>
                <h3 className="font-bold text-base mb-2">{f.title}</h3>
                <p className={`text-sm ${textMuted} leading-relaxed`}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Models section */}
      <section className={`py-20 px-6 ${isDark ? 'bg-[#0d0d0d]' : 'bg-white'}`}>
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-black tracking-tight mb-4">Works with the best models</h2>
          <p className={`${textMuted} mb-10`}>Switch between providers anytime — no lock-in.</p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            {MODELS.map((m, i) => (
              <div key={i} className={`flex items-center gap-3 px-5 py-3 rounded-2xl border ${m.color} text-sm font-semibold`}>
                <Cpu size={16} />
                <div className="text-left">
                  <div className="font-bold">{m.name}</div>
                  <div className="text-[10px] opacity-70 uppercase tracking-wider">{m.provider}</div>
                </div>
              </div>
            ))}
            <div className={`flex items-center gap-2 px-5 py-3 rounded-2xl border ${isDark ? 'border-[#262626] text-[#525252]' : 'border-[#e5e5e5] text-[#a3a3a3]'} text-sm`}>
              <Layers size={16} />
              <span>+ More coming</span>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-black tracking-tight mb-4">How it works</h2>
          </div>
          <div className="space-y-6">
            {[
              { step: '01', title: 'Describe your task', desc: 'Type what you want built, fixed, or automated — in plain English.', icon: <Code2 size={20} /> },
              { step: '02', title: 'Agent plans & executes', desc: 'Mini-Devin breaks the task into steps, writes code, runs tests, and fixes errors autonomously.', icon: <Bot size={20} /> },
              { step: '03', title: 'Review & ship', desc: 'See every action in real time. Approve the result and let the agent commit to GitHub.', icon: <GitBranch size={20} /> },
            ].map((item, i) => (
              <div key={i} className={`flex items-start gap-6 p-6 rounded-2xl border ${cardBg}`}>
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 rounded-2xl bg-[#00ff99]/10 border border-[#00ff99]/20 flex items-center justify-center text-[#00ff99]">
                    {item.icon}
                  </div>
                </div>
                <div className="flex-1">
                  <div className="text-[10px] font-black uppercase tracking-[0.2em] text-[#00ff99]/60 mb-1">Step {item.step}</div>
                  <h3 className="text-lg font-bold mb-1">{item.title}</h3>
                  <p className={`text-sm ${textMuted} leading-relaxed`}>{item.desc}</p>
                </div>
                {i < 2 && <ChevronRight size={16} className="text-[#2a2a2a] flex-shrink-0 mt-4 hidden sm:block" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick start */}
      <section className={`py-20 px-6 ${isDark ? 'bg-[#0d0d0d]' : 'bg-white'}`}>
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-black tracking-tight mb-8 text-center">Quick Start</h2>
          <div className={`rounded-2xl border overflow-hidden ${isDark ? 'border-[#1a1a1a]' : 'border-[#e5e5e5]'}`}>
            <div className={`px-4 py-3 ${isDark ? 'bg-[#161616] border-b border-[#1a1a1a]' : 'bg-[#f5f5f5] border-b border-[#e5e5e5]'} flex items-center gap-2`}>
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
              <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
              <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
              <span className={`ml-2 text-xs font-mono ${textMuted}`}>terminal</span>
            </div>
            <pre className={`p-6 text-sm font-mono overflow-x-auto ${isDark ? 'bg-[#0a0a0a] text-[#c0c0c0]' : 'bg-[#fafafa] text-[#333]'}`}>
{`# Clone the repo
git clone https://github.com/naimprince010-ship-it/mini-devin
cd mini-devin

# Set your API key
echo "OPENAI_API_KEY=your_key" > .env

# Install & run
pip install -r requirements.txt
python -m mini_devin.api.app`}
            </pre>
          </div>
        </div>
      </section>

      {/* CTA Banner */}
      <section className="py-24 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 mb-6 text-[#00ff99]">
            {[...Array(5)].map((_, i) => (
              <Star key={i} size={16} fill="currentColor" />
            ))}
          </div>
          <h2 className="text-3xl sm:text-5xl font-black tracking-tight mb-6">
            Ready to ship faster?
          </h2>
          <p className={`text-lg ${textMuted} mb-10 max-w-lg mx-auto`}>
            Join developers using Mini-Devin to automate repetitive engineering work.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate('/app')}
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-10 py-4 bg-[#00ff99] text-[#0f0f0f] font-black text-lg rounded-2xl hover:bg-[#00e589] hover:scale-105 transition-all shadow-2xl shadow-[#00ff99]/20"
            >
              Launch Mini-Devin
              <Zap size={20} fill="currentColor" />
            </button>
          </div>
          <div className="flex items-center justify-center gap-6 mt-8">
            {['Free to self-host', 'Open source', 'No credit card needed'].map((t, i) => (
              <div key={i} className={`flex items-center gap-1.5 text-xs ${textMuted}`}>
                <CheckCircle2 size={12} className="text-[#00ff99]" />
                {t}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className={`border-t ${isDark ? 'border-[#1a1a1a]' : 'border-[#e5e5e5]'} py-8 px-6`}>
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Bot size={16} className="text-[#00ff99]" />
            <span className="font-bold text-sm">Mini-Devin</span>
            <span className={`text-xs ${textMuted}`}>— Autonomous AI Software Engineer</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="/docs" className={`text-xs ${textMuted} hover:text-[#00ff99] transition-colors`}>API Docs</a>
            <a
              href="https://github.com/naimprince010-ship-it/mini-devin"
              target="_blank"
              className={`flex items-center gap-1.5 text-xs ${textMuted} hover:text-[#00ff99] transition-colors`}
            >
              <Github size={13} />
              GitHub
            </a>
            <button
              onClick={() => navigate('/app')}
              className="flex items-center gap-1.5 text-xs text-[#00ff99] font-bold hover:underline"
            >
              Launch App
              <ArrowRight size={12} />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
