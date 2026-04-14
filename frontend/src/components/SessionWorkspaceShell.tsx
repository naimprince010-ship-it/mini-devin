import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import {
  Terminal,
  History,
  ListTree,
  FileCode,
  Globe,
  Brain,
  PanelRightClose,
  PanelRightOpen,
} from 'lucide-react';
import type { Session } from '../types';
import { TaskPanel } from './TaskPanel';
import { WorkspacePanel, type WorkspacePanelTab } from './WorkspacePanel';
import { ErrorBoundary } from './ErrorBoundary';

const WORKSPACE_PANEL_LS = 'plodder:workspace-panel-open';

type ShellProps = {
  session: Session;
  isDark: boolean;
  onTitleUpdated: (title: string) => void;
  onSessionUpdated: (s: Session) => void;
};

const TAB_ICONS: Record<WorkspacePanelTab, ReactNode> = {
  shell: <Terminal className="h-5 w-5" />,
  worklog: <History className="h-5 w-5" />,
  feed: <ListTree className="h-5 w-5" />,
  editor: <FileCode className="h-5 w-5" />,
  browser: <Globe className="h-5 w-5" />,
  memory: <Brain className="h-5 w-5" />,
};

const TAB_LABELS: Record<WorkspacePanelTab, string> = {
  shell: 'Shell',
  worklog: 'Worklog',
  feed: 'Activity',
  editor: 'IDE',
  browser: 'Browser',
  memory: 'Memory',
};

const ALL_TABS: WorkspacePanelTab[] = [
  'shell',
  'worklog',
  'feed',
  'editor',
  'browser',
  'memory',
];

function loadPanelOpen(): boolean {
  try {
    const v = localStorage.getItem(WORKSPACE_PANEL_LS);
    if (v === '0') return false;
    return true;
  } catch {
    return true;
  }
}

export function SessionWorkspaceShell({
  session,
  isDark,
  onTitleUpdated,
  onSessionUpdated,
}: ShellProps) {
  const [workspaceTab, setWorkspaceTab] = useState<WorkspacePanelTab>('shell');
  const [workspaceOpen, setWorkspaceOpen] = useState(loadPanelOpen);
  const [mobilePane, setMobilePane] = useState<'chat' | 'workspace'>('chat');

  useEffect(() => {
    setWorkspaceTab('shell');
    setMobilePane('chat');
  }, [session.session_id]);

  const persistOpen = useCallback((open: boolean) => {
    setWorkspaceOpen(open);
    try {
      localStorage.setItem(WORKSPACE_PANEL_LS, open ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, []);

  const tabIdle = isDark
    ? 'text-[#9299AA] bg-[#0D0F11]'
    : 'text-[#64748b] bg-[#f1f5f9]';
  const tabActive = isDark
    ? 'bg-[#25272D] text-white'
    : 'bg-[#e2e8f0] text-[#0f172a]';
  const handleBg = isDark ? 'hover:bg-[#1a1d22]' : 'hover:bg-[#e8eef4]';

  const wd = session.working_directory?.trim();
  const wdShort =
    wd && wd.length > 0
      ? wd.replace(/\\/g, '/').split('/').filter(Boolean).slice(-2).join('/')
      : null;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col gap-2 p-2 md:gap-3 md:p-0">
      {/* OpenHands-style: conversation identity (left) + workspace tab strip (right) */}
      <div className="flex flex-shrink-0 flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0 flex-1">
          <h1
            className={`truncate text-base font-semibold tracking-tight ${
              isDark ? 'text-white' : 'text-[#0f172a]'
            }`}
          >
            {session.title || 'Session'}
          </h1>
          {wdShort ? (
            <p
              className={`mt-0.5 truncate font-mono text-[11px] ${
                isDark ? 'text-[#737373]' : 'text-[#64748b]'
              }`}
              title={wd}
            >
              {wdShort}
            </p>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center justify-start gap-1.5 sm:justify-end md:gap-2">
          {ALL_TABS.map((id) => {
            const active = workspaceTab === id;
            return (
              <button
                key={id}
                type="button"
                data-testid={`workspace-tab-${id}`}
                onClick={() => {
                  setWorkspaceTab(id);
                  persistOpen(true);
                  setMobilePane('workspace');
                }}
                title={TAB_LABELS[id]}
                className={`flex cursor-pointer items-center gap-2 rounded-md pl-1.5 pr-2 py-1 transition-colors md:py-1.5 ${tabIdle} ${handleBg} ${
                  active ? tabActive : ''
                }`}
              >
                <span className="flex-shrink-0 [&>svg]:text-inherit">
                  {TAB_ICONS[id]}
                </span>
                {active ? (
                  <span className="whitespace-nowrap text-sm font-medium">
                    {TAB_LABELS[id]}
                  </span>
                ) : null}
              </button>
            );
          })}

          <button
            type="button"
            title={workspaceOpen ? 'Hide workspace panel' : 'Show workspace panel'}
            aria-label={workspaceOpen ? 'Hide workspace panel' : 'Show workspace panel'}
            onClick={() => persistOpen(!workspaceOpen)}
            className={`rounded-md p-1.5 ${tabIdle} ${handleBg} hidden md:inline-flex`}
          >
            {workspaceOpen ? (
              <PanelRightClose className="h-5 w-5" />
            ) : (
              <PanelRightOpen className="h-5 w-5" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile: chat vs workspace (OpenHands-style stacked / sheet intent) */}
      <div className="flex flex-shrink-0 items-center gap-1 rounded-lg border border-[var(--border-color)] p-0.5 md:hidden">
        <button
          type="button"
          className={`flex-1 rounded-md py-1.5 text-xs font-medium ${
            mobilePane === 'chat'
              ? isDark
                ? 'bg-[#25272D] text-white'
                : 'bg-[#e2e8f0] text-[#0f172a]'
              : isDark
                ? 'text-[#9299AA]'
                : 'text-[#64748b]'
          }`}
          onClick={() => setMobilePane('chat')}
        >
          Chat
        </button>
        <button
          type="button"
          className={`flex-1 rounded-md py-1.5 text-xs font-medium ${
            mobilePane === 'workspace'
              ? isDark
                ? 'bg-[#25272D] text-white'
                : 'bg-[#e2e8f0] text-[#0f172a]'
              : isDark
                ? 'text-[#9299AA]'
                : 'text-[#64748b]'
          }`}
          onClick={() => setMobilePane('workspace')}
        >
          Workspace
        </button>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        {/* Desktop: resizable split */}
        <div
          className={`hidden h-full min-h-0 md:block ${!workspaceOpen ? 'flex flex-col' : ''}`}
        >
          {workspaceOpen ? (
            <PanelGroup direction="horizontal" className="h-full">
              <Panel defaultSize={50} minSize={28} maxSize={72}>
                <div className="flex h-full min-h-0 flex-col overflow-hidden">
                  <ErrorBoundary>
                    <TaskPanel
                      session={session}
                      onTitleUpdated={onTitleUpdated}
                      onSessionUpdated={onSessionUpdated}
                      workspaceChrome="embedded"
                    />
                  </ErrorBoundary>
                </div>
              </Panel>
              <PanelResizeHandle
                className={`w-px flex-shrink-0 cursor-col-resize transition-colors hover:bg-[#00ff99]/30 ${isDark ? 'bg-[#1a1a1a]' : 'bg-[#e5e5e5]'}`}
              />
              <Panel defaultSize={50} minSize={28}>
                <ErrorBoundary>
                  <WorkspacePanel
                    sessionId={session.session_id}
                    activeTab={workspaceTab}
                    onActiveTabChange={setWorkspaceTab}
                    tabBar="external"
                    frameStyle="openhands"
                    borderTone={isDark ? 'dark' : 'light'}
                  />
                </ErrorBoundary>
              </Panel>
            </PanelGroup>
          ) : (
            <div className="h-full min-h-0">
              <ErrorBoundary>
                <TaskPanel
                  session={session}
                  onTitleUpdated={onTitleUpdated}
                  onSessionUpdated={onSessionUpdated}
                  workspaceChrome="embedded"
                />
              </ErrorBoundary>
            </div>
          )}
        </div>

        {/* Mobile: single pane */}
        <div className="flex h-full min-h-0 flex-col md:hidden">
          {mobilePane === 'chat' ? (
            <ErrorBoundary>
              <TaskPanel
                session={session}
                onTitleUpdated={onTitleUpdated}
                onSessionUpdated={onSessionUpdated}
                workspaceChrome="embedded"
              />
            </ErrorBoundary>
          ) : (
            <ErrorBoundary>
              <WorkspacePanel
                sessionId={session.session_id}
                activeTab={workspaceTab}
                onActiveTabChange={setWorkspaceTab}
                tabBar="external"
                frameStyle="openhands"
                borderTone={isDark ? 'dark' : 'light'}
              />
            </ErrorBoundary>
          )}
        </div>
      </div>
    </div>
  );
}
