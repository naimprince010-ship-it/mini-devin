import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="h-full flex flex-col items-center justify-center bg-[#0f0f0f] text-white p-8">
          <div className="max-w-md w-full bg-[#111111] border border-red-500/20 rounded-2xl p-8 text-center space-y-5">
            <div className="w-16 h-16 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto">
              <AlertTriangle className="text-red-400" size={28} />
            </div>

            <div>
              <h2 className="text-lg font-semibold text-white mb-1">Something went wrong</h2>
              <p className="text-sm text-[#a3a3a3]">
                An unexpected error occurred. You can try resetting this section or reload the page.
              </p>
            </div>

            {this.state.error && (
              <details className="text-left">
                <summary className="text-xs text-[#737373] cursor-pointer hover:text-[#a3a3a3] transition-colors">
                  Error details
                </summary>
                <pre className="mt-2 p-3 bg-[#1a1a1a] rounded-lg text-xs text-red-400 overflow-auto max-h-32 whitespace-pre-wrap">
                  {this.state.error.message}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}

            <div className="flex gap-3">
              <button
                onClick={this.handleReset}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg border border-[#262626] text-sm text-[#a3a3a3] hover:text-white hover:border-[#363636] transition-colors"
              >
                <Home size={14} />
                Reset
              </button>
              <button
                onClick={this.handleReload}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg bg-[#00ff99] text-[#0f0f0f] text-sm font-semibold hover:bg-[#00e589] transition-colors"
              >
                <RefreshCw size={14} />
                Reload Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
