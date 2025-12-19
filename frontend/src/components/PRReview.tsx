import { useState, useCallback } from 'react';
import { GitPullRequest, Search, Send, CheckCircle, XCircle, MessageSquare, AlertTriangle, Info, Loader2, ChevronDown, ChevronUp, Code, FileCode } from 'lucide-react';

interface ReviewFinding {
  id: string;
  path: string;
  line: number;
  severity: 'error' | 'warning' | 'suggestion' | 'info';
  category: string;
  message: string;
  suggestion?: string;
  confidence: number;
}

interface ReviewResult {
  pr: {
    owner: string;
    repo: string;
    number: number;
    head_sha: string;
  };
  summary: string;
  verdict: 'approve' | 'request_changes' | 'comment_only';
  findings: ReviewFinding[];
  analyzed_at: string;
  model_used: string;
}

interface PRContext {
  success: boolean;
  error?: string;
  pr: {
    owner: string;
    repo: string;
    number: number;
    title: string;
    body: string;
    author: string;
    head_sha: string;
    base_branch: string;
    head_branch: string;
    html_url: string;
  };
  files: Array<{
    filename: string;
    status: string;
    additions: number;
    deletions: number;
  }>;
  total_additions: number;
  total_deletions: number;
}

interface PRReviewProps {
  apiBaseUrl?: string;
}

export function PRReview({ apiBaseUrl = 'http://localhost:8000/api' }: PRReviewProps) {
  const [owner, setOwner] = useState('');
  const [repo, setRepo] = useState('');
  const [prNumber, setPrNumber] = useState('');
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prContext, setPrContext] = useState<PRContext | null>(null);
  const [reviewResult, setReviewResult] = useState<ReviewResult | null>(null);
  const [expandedFindings, setExpandedFindings] = useState<Set<string>>(new Set());
  const [focusAreas, setFocusAreas] = useState<string[]>([]);
  const [includeSuggestions, setIncludeSuggestions] = useState(true);

  const getAuthHeaders = useCallback(() => {
    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  }, []);

  const handleFetchPR = async () => {
    if (!owner || !repo || !prNumber) {
      setError('Please enter owner, repo, and PR number');
      return;
    }

    setLoading(true);
    setError(null);
    setPrContext(null);
    setReviewResult(null);

    try {
      const response = await fetch(
        `${apiBaseUrl}/github/prs/${owner}/${repo}/${prNumber}/context`,
        { headers: getAuthHeaders() }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch PR');
      }

      const data = await response.json();
      setPrContext(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch PR');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzePR = async () => {
    if (!prContext) return;

    setAnalyzing(true);
    setError(null);

    try {
      const response = await fetch(
        `${apiBaseUrl}/github/prs/${owner}/${repo}/${prNumber}/review/analyze`,
        {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({
            include_suggestions: includeSuggestions,
            focus_areas: focusAreas.length > 0 ? focusAreas : null,
            max_files: 50
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze PR');
      }

      const data = await response.json();
      setReviewResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to analyze PR');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSubmitReview = async (event: 'APPROVE' | 'REQUEST_CHANGES' | 'COMMENT') => {
    if (!reviewResult) return;

    setSubmitting(true);
    setError(null);

    try {
      const response = await fetch(
        `${apiBaseUrl}/github/prs/${owner}/${repo}/${prNumber}/review/submit`,
        {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({
            review_result: reviewResult,
            event,
            include_suggestions: includeSuggestions
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to submit review');
      }

      const data = await response.json();
      alert(`Review submitted successfully! View at: ${data.html_url || 'GitHub'}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to submit review');
    } finally {
      setSubmitting(false);
    }
  };

  const handleQuickReview = async () => {
    if (!owner || !repo || !prNumber) {
      setError('Please enter owner, repo, and PR number');
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      const response = await fetch(
        `${apiBaseUrl}/github/prs/${owner}/${repo}/${prNumber}/review/quick`,
        {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({
            include_suggestions: includeSuggestions,
            focus_areas: focusAreas.length > 0 ? focusAreas : null,
            max_files: 50
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to review PR');
      }

      const data = await response.json();
      setReviewResult(data.review);
      alert(`Review submitted! View at: ${data.submission?.html_url || 'GitHub'}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to review PR');
    } finally {
      setAnalyzing(false);
    }
  };

  const toggleFinding = (id: string) => {
    setExpandedFindings(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <XCircle size={16} className="text-red-400" />;
      case 'warning':
        return <AlertTriangle size={16} className="text-yellow-400" />;
      case 'suggestion':
        return <MessageSquare size={16} className="text-blue-400" />;
      case 'info':
        return <Info size={16} className="text-gray-400" />;
      default:
        return <Info size={16} className="text-gray-400" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error':
        return 'border-red-500/50 bg-red-900/20';
      case 'warning':
        return 'border-yellow-500/50 bg-yellow-900/20';
      case 'suggestion':
        return 'border-blue-500/50 bg-blue-900/20';
      default:
        return 'border-gray-500/50 bg-gray-900/20';
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case 'approve':
        return <CheckCircle size={20} className="text-green-400" />;
      case 'request_changes':
        return <XCircle size={20} className="text-red-400" />;
      default:
        return <MessageSquare size={20} className="text-blue-400" />;
    }
  };

  const focusAreaOptions = ['security', 'performance', 'style', 'logic', 'best_practice', 'bug'];

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <GitPullRequest size={24} className="text-purple-400" />
        <h2 className="text-lg font-semibold text-white">PR Code Review</h2>
      </div>

      {error && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-300 hover:text-red-100">
            Dismiss
          </button>
        </div>
      )}

      {/* PR Input Form */}
      <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
        <h3 className="text-sm font-medium text-white mb-3">Select Pull Request</h3>
        <div className="grid grid-cols-3 gap-3 mb-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Owner</label>
            <input
              type="text"
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
              placeholder="owner"
              className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Repository</label>
            <input
              type="text"
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="repo"
              className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">PR Number</label>
            <input
              type="number"
              value={prNumber}
              onChange={(e) => setPrNumber(e.target.value)}
              placeholder="123"
              className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
        </div>

        {/* Options */}
        <div className="mb-3">
          <label className="block text-xs text-gray-400 mb-2">Focus Areas (optional)</label>
          <div className="flex flex-wrap gap-2">
            {focusAreaOptions.map((area) => (
              <button
                key={area}
                onClick={() => {
                  setFocusAreas(prev =>
                    prev.includes(area)
                      ? prev.filter(a => a !== area)
                      : [...prev, area]
                  );
                }}
                className={`px-2 py-1 text-xs rounded-lg transition-colors ${
                  focusAreas.includes(area)
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {area}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 mb-3">
          <input
            type="checkbox"
            id="includeSuggestions"
            checked={includeSuggestions}
            onChange={(e) => setIncludeSuggestions(e.target.checked)}
            className="rounded bg-gray-700 border-gray-600"
          />
          <label htmlFor="includeSuggestions" className="text-sm text-gray-300">
            Include code suggestions (GitHub suggestion blocks)
          </label>
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleFetchPR}
            disabled={loading || !owner || !repo || !prNumber}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Search size={16} />
            )}
            Fetch PR
          </button>
          <button
            onClick={handleQuickReview}
            disabled={analyzing || !owner || !repo || !prNumber}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
          >
            {analyzing ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
            Quick Review & Submit
          </button>
        </div>
      </div>

      {/* PR Context Display */}
      {prContext && (
        <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
          <div className="flex items-start justify-between mb-3">
            <div>
              <h3 className="text-white font-medium">
                #{prContext.pr.number}: {prContext.pr.title}
              </h3>
              <p className="text-gray-400 text-sm">
                {prContext.pr.author} wants to merge {prContext.pr.head_branch} into {prContext.pr.base_branch}
              </p>
            </div>
            <a
              href={prContext.pr.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-purple-400 hover:text-purple-300 text-sm"
            >
              View on GitHub
            </a>
          </div>

          <div className="flex items-center gap-4 text-sm mb-3">
            <span className="text-green-400">+{prContext.total_additions}</span>
            <span className="text-red-400">-{prContext.total_deletions}</span>
            <span className="text-gray-400">{prContext.files.length} files changed</span>
          </div>

          {/* Files List */}
          <div className="mb-3">
            <h4 className="text-xs text-gray-400 mb-2">Changed Files</h4>
            <div className="max-h-40 overflow-y-auto space-y-1">
              {prContext.files.map((file, idx) => (
                <div key={idx} className="flex items-center gap-2 text-sm">
                  <FileCode size={14} className="text-gray-500" />
                  <span className="text-gray-300 truncate">{file.filename}</span>
                  <span className="text-green-400 text-xs">+{file.additions}</span>
                  <span className="text-red-400 text-xs">-{file.deletions}</span>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={handleAnalyzePR}
            disabled={analyzing}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
          >
            {analyzing ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Code size={16} />
                Analyze with AI
              </>
            )}
          </button>
        </div>
      )}

      {/* Review Results */}
      {reviewResult && (
        <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            {getVerdictIcon(reviewResult.verdict)}
            <h3 className="text-white font-medium">
              Review Result: {reviewResult.verdict.replace('_', ' ').toUpperCase()}
            </h3>
          </div>

          <p className="text-gray-300 text-sm mb-4">{reviewResult.summary}</p>

          <div className="text-xs text-gray-500 mb-4">
            Analyzed at {new Date(reviewResult.analyzed_at).toLocaleString()} using {reviewResult.model_used}
          </div>

          {/* Findings */}
          {reviewResult.findings.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm text-gray-400 mb-2">
                Findings ({reviewResult.findings.length})
              </h4>
              <div className="space-y-2">
                {reviewResult.findings.map((finding) => (
                  <div
                    key={finding.id}
                    className={`border rounded-lg ${getSeverityColor(finding.severity)}`}
                  >
                    <div
                      onClick={() => toggleFinding(finding.id)}
                      className="flex items-center justify-between p-3 cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        {getSeverityIcon(finding.severity)}
                        <span className="text-white text-sm">{finding.path}:{finding.line}</span>
                        <span className="text-gray-400 text-xs">({finding.category})</span>
                      </div>
                      {expandedFindings.has(finding.id) ? (
                        <ChevronUp size={16} className="text-gray-400" />
                      ) : (
                        <ChevronDown size={16} className="text-gray-400" />
                      )}
                    </div>
                    {expandedFindings.has(finding.id) && (
                      <div className="px-3 pb-3 border-t border-gray-700">
                        <p className="text-gray-300 text-sm mt-2">{finding.message}</p>
                        {finding.suggestion && (
                          <div className="mt-2">
                            <span className="text-xs text-gray-400">Suggested fix:</span>
                            <pre className="mt-1 p-2 bg-gray-900 rounded text-green-400 text-xs overflow-x-auto">
                              {finding.suggestion}
                            </pre>
                          </div>
                        )}
                        <div className="mt-2 text-xs text-gray-500">
                          Confidence: {Math.round(finding.confidence * 100)}%
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Submit Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => handleSubmitReview('APPROVE')}
              disabled={submitting}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
            >
              {submitting ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle size={16} />}
              Approve
            </button>
            <button
              onClick={() => handleSubmitReview('REQUEST_CHANGES')}
              disabled={submitting}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
            >
              {submitting ? <Loader2 size={16} className="animate-spin" /> : <XCircle size={16} />}
              Request Changes
            </button>
            <button
              onClick={() => handleSubmitReview('COMMENT')}
              disabled={submitting}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
            >
              {submitting ? <Loader2 size={16} className="animate-spin" /> : <MessageSquare size={16} />}
              Comment Only
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
