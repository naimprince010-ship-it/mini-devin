import { useState, useEffect } from 'react';
import { Plus, Trash2, RefreshCw, Play, Edit2, X, Tag, ChevronDown, ChevronRight } from 'lucide-react';

interface SkillParameter {
  name: string;
  description: string;
  type: string;
  required: boolean;
  default?: string | number | boolean;
}

interface SkillStep {
  name: string;
  description: string;
  tool: string;
  args?: Record<string, string>;
}

interface Skill {
  name: string;
  description: string;
  version: string;
  tags: string[];
  required_tools: string[];
  parameters: SkillParameter[];
  steps?: SkillStep[];
  is_custom: boolean;
}

interface SkillsManagerProps {
  apiBaseUrl?: string;
}

export function SkillsManager({ apiBaseUrl = 'http://localhost:8000/api' }: SkillsManagerProps) {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingSkill, setEditingSkill] = useState<Skill | null>(null);
  const [expandedSkill, setExpandedSkill] = useState<string | null>(null);
  const [executeParams, setExecuteParams] = useState<Record<string, string>>({});
  const [executingSkill, setExecutingSkill] = useState<string | null>(null);

  const [newSkill, setNewSkill] = useState({
    name: '',
    description: '',
    tags: [] as string[],
    parameters: [] as SkillParameter[],
    steps: [] as SkillStep[],
  });

  const loadSkills = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (selectedTag) params.append('tag', selectedTag);
      if (searchQuery) params.append('search', searchQuery);
      
      const response = await fetch(`${apiBaseUrl}/skills?${params}`);
      if (!response.ok) throw new Error('Failed to load skills');
      const data = await response.json();
      setSkills(data.skills);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load skills');
    } finally {
      setLoading(false);
    }
  };

  const loadTags = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/skills/tags`);
      if (!response.ok) throw new Error('Failed to load tags');
      const data = await response.json();
      setTags(data.tags);
    } catch (e) {
      console.error('Failed to load tags:', e);
    }
  };

  useEffect(() => {
    loadSkills();
    loadTags();
  }, [selectedTag, searchQuery]);

  const handleCreateSkill = async () => {
    if (!newSkill.name || !newSkill.description) {
      setError('Name and description are required');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/skills`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSkill),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create skill');
      }
      
      await loadSkills();
      await loadTags();
      setShowCreateForm(false);
      setNewSkill({ name: '', description: '', tags: [], parameters: [], steps: [] });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create skill');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateSkill = async () => {
    if (!editingSkill) return;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/skills/${editingSkill.name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: editingSkill.description,
          tags: editingSkill.tags,
          parameters: editingSkill.parameters,
          steps: editingSkill.steps,
        }),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to update skill');
      }
      
      await loadSkills();
      await loadTags();
      setEditingSkill(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to update skill');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSkill = async (skillName: string) => {
    if (!confirm(`Are you sure you want to delete "${skillName}"?`)) return;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/skills/${skillName}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to delete skill');
      }
      
      await loadSkills();
      await loadTags();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete skill');
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteSkill = async (skillName: string, dryRun: boolean = false) => {
    setExecutingSkill(skillName);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/skills/${skillName}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: executeParams,
          workspace_path: '.',
          dry_run: dryRun,
        }),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to execute skill');
      }
      
      const data = await response.json();
      alert(`Skill execution started! ID: ${data.execution_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to execute skill');
    } finally {
      setExecutingSkill(null);
    }
  };

  const addParameter = (isNew: boolean) => {
    const param: SkillParameter = {
      name: '',
      description: '',
      type: 'string',
      required: true,
    };
    
    if (isNew) {
      setNewSkill({ ...newSkill, parameters: [...newSkill.parameters, param] });
    } else if (editingSkill) {
      setEditingSkill({ ...editingSkill, parameters: [...editingSkill.parameters, param] });
    }
  };

  const addStep = (isNew: boolean) => {
    const step: SkillStep = {
      name: '',
      description: '',
      tool: 'terminal',
    };
    
    if (isNew) {
      setNewSkill({ ...newSkill, steps: [...newSkill.steps, step] });
    } else if (editingSkill) {
      setEditingSkill({ ...editingSkill, steps: [...(editingSkill.steps || []), step] });
    }
  };

  const removeParameter = (index: number, isNew: boolean) => {
    if (isNew) {
      setNewSkill({ ...newSkill, parameters: newSkill.parameters.filter((_, i) => i !== index) });
    } else if (editingSkill) {
      setEditingSkill({ ...editingSkill, parameters: editingSkill.parameters.filter((_, i) => i !== index) });
    }
  };

  const removeStep = (index: number, isNew: boolean) => {
    if (isNew) {
      setNewSkill({ ...newSkill, steps: newSkill.steps.filter((_, i) => i !== index) });
    } else if (editingSkill) {
      setEditingSkill({ ...editingSkill, steps: (editingSkill.steps || []).filter((_, i) => i !== index) });
    }
  };

  const renderSkillForm = (skill: typeof newSkill | Skill, isNew: boolean) => {
    const updateField = (field: string, value: string | string[]) => {
      if (isNew) {
        setNewSkill({ ...newSkill, [field]: value });
      } else if (editingSkill) {
        setEditingSkill({ ...editingSkill, [field]: value });
      }
    };

    const updateParameter = (index: number, field: string, value: string | boolean) => {
      const params = isNew ? [...newSkill.parameters] : [...(editingSkill?.parameters || [])];
      params[index] = { ...params[index], [field]: value };
      if (isNew) {
        setNewSkill({ ...newSkill, parameters: params });
      } else if (editingSkill) {
        setEditingSkill({ ...editingSkill, parameters: params });
      }
    };

    const updateStep = (index: number, field: string, value: string) => {
      const steps = isNew ? [...newSkill.steps] : [...(editingSkill?.steps || [])];
      steps[index] = { ...steps[index], [field]: value };
      if (isNew) {
        setNewSkill({ ...newSkill, steps: steps });
      } else if (editingSkill) {
        setEditingSkill({ ...editingSkill, steps: steps });
      }
    };

    const params = isNew ? newSkill.parameters : (editingSkill?.parameters || []);
    const steps = isNew ? newSkill.steps : (editingSkill?.steps || []);

    return (
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Name</label>
          <input
            type="text"
            value={skill.name}
            onChange={(e) => updateField('name', e.target.value)}
            disabled={!isNew}
            className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
            placeholder="my_custom_skill"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-300 mb-1">Description</label>
          <textarea
            value={skill.description}
            onChange={(e) => updateField('description', e.target.value)}
            className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
            placeholder="What does this skill do?"
            rows={2}
          />
        </div>

        <div>
          <label className="block text-sm text-gray-300 mb-1">Tags (comma-separated)</label>
          <input
            type="text"
            value={skill.tags.join(', ')}
            onChange={(e) => updateField('tags', e.target.value.split(',').map(t => t.trim()).filter(Boolean))}
            className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
            placeholder="automation, testing, deployment"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-gray-300">Parameters</label>
            <button
              onClick={() => addParameter(isNew)}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              + Add Parameter
            </button>
          </div>
          {params.map((param, index) => (
            <div key={index} className="flex gap-2 mb-2 p-2 bg-gray-700 rounded">
              <input
                type="text"
                value={param.name}
                onChange={(e) => updateParameter(index, 'name', e.target.value)}
                className="flex-1 px-2 py-1 bg-gray-600 text-white rounded text-sm"
                placeholder="param_name"
              />
              <select
                value={param.type}
                onChange={(e) => updateParameter(index, 'type', e.target.value)}
                className="px-2 py-1 bg-gray-600 text-white rounded text-sm"
              >
                <option value="string">string</option>
                <option value="integer">integer</option>
                <option value="boolean">boolean</option>
                <option value="array">array</option>
              </select>
              <label className="flex items-center text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={param.required}
                  onChange={(e) => updateParameter(index, 'required', e.target.checked)}
                  className="mr-1"
                />
                Required
              </label>
              <button
                onClick={() => removeParameter(index, isNew)}
                className="text-red-400 hover:text-red-300"
              >
                <X size={16} />
              </button>
            </div>
          ))}
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-gray-300">Steps</label>
            <button
              onClick={() => addStep(isNew)}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              + Add Step
            </button>
          </div>
          {steps.map((step, index) => (
            <div key={index} className="mb-2 p-2 bg-gray-700 rounded">
              <div className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={step.name}
                  onChange={(e) => updateStep(index, 'name', e.target.value)}
                  className="flex-1 px-2 py-1 bg-gray-600 text-white rounded text-sm"
                  placeholder="Step name"
                />
                <select
                  value={step.tool}
                  onChange={(e) => updateStep(index, 'tool', e.target.value)}
                  className="px-2 py-1 bg-gray-600 text-white rounded text-sm"
                >
                  <option value="terminal">terminal</option>
                  <option value="editor">editor</option>
                  <option value="browser_search">browser_search</option>
                  <option value="browser_fetch">browser_fetch</option>
                </select>
                <button
                  onClick={() => removeStep(index, isNew)}
                  className="text-red-400 hover:text-red-300"
                >
                  <X size={16} />
                </button>
              </div>
              <input
                type="text"
                value={step.description}
                onChange={(e) => updateStep(index, 'description', e.target.value)}
                className="w-full px-2 py-1 bg-gray-600 text-white rounded text-sm"
                placeholder="Step description"
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Skills</h2>
        <div className="flex gap-2">
          <button
            onClick={loadSkills}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Refresh"
          >
            <RefreshCw size={16} />
          </button>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="New Skill"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-500 rounded text-red-200 text-sm">
          {error}
          <button onClick={() => setError(null)} className="float-right text-red-400 hover:text-red-300">
            <X size={14} />
          </button>
        </div>
      )}

      <div className="mb-4 flex gap-2">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex-1 px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
          placeholder="Search skills..."
        />
        <select
          value={selectedTag || ''}
          onChange={(e) => setSelectedTag(e.target.value || null)}
          className="px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
        >
          <option value="">All Tags</option>
          {tags.map(tag => (
            <option key={tag} value={tag}>{tag}</option>
          ))}
        </select>
      </div>

      {showCreateForm && (
        <div className="mb-4 p-4 bg-gray-700 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-white font-medium">Create New Skill</h3>
            <button onClick={() => setShowCreateForm(false)} className="text-gray-400 hover:text-white">
              <X size={16} />
            </button>
          </div>
          {renderSkillForm(newSkill, true)}
          <div className="mt-4 flex gap-2">
            <button
              onClick={handleCreateSkill}
              disabled={loading}
              className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium disabled:opacity-50"
            >
              {loading ? 'Creating...' : 'Create Skill'}
            </button>
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {editingSkill && (
        <div className="mb-4 p-4 bg-gray-700 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-white font-medium">Edit Skill: {editingSkill.name}</h3>
            <button onClick={() => setEditingSkill(null)} className="text-gray-400 hover:text-white">
              <X size={16} />
            </button>
          </div>
          {renderSkillForm(editingSkill, false)}
          <div className="mt-4 flex gap-2">
            <button
              onClick={handleUpdateSkill}
              disabled={loading}
              className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium disabled:opacity-50"
            >
              {loading ? 'Saving...' : 'Save Changes'}
            </button>
            <button
              onClick={() => setEditingSkill(null)}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {loading && skills.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-4">Loading skills...</p>
        ) : skills.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-4">No skills found</p>
        ) : (
          skills.map((skill) => (
            <div
              key={skill.name}
              className="bg-gray-700 rounded-lg overflow-hidden"
            >
              <div
                onClick={() => setExpandedSkill(expandedSkill === skill.name ? null : skill.name)}
                className="p-3 cursor-pointer hover:bg-gray-600 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {expandedSkill === skill.name ? (
                      <ChevronDown size={16} className="text-gray-400" />
                    ) : (
                      <ChevronRight size={16} className="text-gray-400" />
                    )}
                    <span className="text-white font-medium">{skill.name}</span>
                    {skill.is_custom && (
                      <span className="px-2 py-0.5 bg-blue-600 text-xs text-white rounded">Custom</span>
                    )}
                    <span className="text-gray-400 text-xs">v{skill.version}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    {skill.is_custom && (
                      <>
                        <button
                          onClick={(e) => { e.stopPropagation(); setEditingSkill(skill); }}
                          className="p-1 text-gray-400 hover:text-blue-400"
                          title="Edit"
                        >
                          <Edit2 size={14} />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDeleteSkill(skill.name); }}
                          className="p-1 text-gray-400 hover:text-red-400"
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      </>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleExecuteSkill(skill.name, false); }}
                      disabled={executingSkill === skill.name}
                      className="p-1 text-gray-400 hover:text-green-400 disabled:opacity-50"
                      title="Execute"
                    >
                      <Play size={14} />
                    </button>
                  </div>
                </div>
                <p className="text-gray-300 text-sm mt-1">{skill.description}</p>
                {skill.tags.length > 0 && (
                  <div className="flex gap-1 mt-2">
                    {skill.tags.map(tag => (
                      <span
                        key={tag}
                        onClick={(e) => { e.stopPropagation(); setSelectedTag(tag); }}
                        className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-600 text-xs text-gray-300 rounded cursor-pointer hover:bg-gray-500"
                      >
                        <Tag size={10} />
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>

                            {expandedSkill === skill.name && (
                              <div className="px-3 pb-3 border-t border-gray-600">
                                <div className="mt-3">
                                  <h4 className="text-sm text-gray-400 mb-2">Required Tools</h4>
                                  <div className="flex flex-wrap gap-1">
                                    {(skill.required_tools ?? []).length > 0 ? (
                                      (skill.required_tools ?? []).map(tool => (
                                        <span key={tool} className="px-2 py-0.5 bg-gray-600 text-xs text-gray-300 rounded">
                                          {tool}
                                        </span>
                                      ))
                                    ) : (
                                      <span className="text-gray-500 text-xs">None</span>
                                    )}
                                  </div>
                                </div>

                                                    {(skill.parameters ?? []).length > 0 && (
                                        <div className="mt-3">
                                          <h4 className="text-sm text-gray-400 mb-2">Parameters</h4>
                                          <div className="space-y-2">
                                            {(skill.parameters ?? []).map((param, index) => (
                          <div key={index} className="flex items-center gap-2">
                            <input
                              type="text"
                              placeholder={`${param.name} (${param.type}${param.required ? ', required' : ''})`}
                              value={executeParams[param.name] || ''}
                              onChange={(e) => setExecuteParams({ ...executeParams, [param.name]: e.target.value })}
                              className="flex-1 px-2 py-1 bg-gray-600 text-white rounded text-sm"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {skill.steps && skill.steps.length > 0 && (
                    <div className="mt-3">
                      <h4 className="text-sm text-gray-400 mb-2">Steps</h4>
                      <ol className="list-decimal list-inside space-y-1">
                        {skill.steps.map((step, index) => (
                          <li key={index} className="text-gray-300 text-sm">
                            <span className="font-medium">{step.name}</span>
                            <span className="text-gray-400"> ({step.tool})</span>
                            {step.description && (
                              <span className="text-gray-500"> - {step.description}</span>
                            )}
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}

                  <div className="mt-3 flex gap-2">
                    <button
                      onClick={() => handleExecuteSkill(skill.name, true)}
                      disabled={executingSkill === skill.name}
                      className="px-3 py-1 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm disabled:opacity-50"
                    >
                      Dry Run
                    </button>
                    <button
                      onClick={() => handleExecuteSkill(skill.name, false)}
                      disabled={executingSkill === skill.name}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm disabled:opacity-50"
                    >
                      {executingSkill === skill.name ? 'Executing...' : 'Execute'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
