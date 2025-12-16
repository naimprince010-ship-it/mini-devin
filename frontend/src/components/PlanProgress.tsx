import { CheckCircle, Circle, Loader, ListTodo } from 'lucide-react';

export interface PlanStep {
  id: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  substeps?: PlanStep[];
}

export interface Plan {
  id: string;
  title: string;
  strategy: string;
  steps: PlanStep[];
  currentStepIndex: number;
  totalSteps: number;
  completedSteps: number;
}

interface PlanProgressProps {
  plan: Plan | null;
}

const getStepIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle size={16} className="text-green-400" />;
    case 'in_progress':
      return <Loader size={16} className="text-blue-400 animate-spin" />;
    case 'skipped':
      return <Circle size={16} className="text-gray-600" />;
    default:
      return <Circle size={16} className="text-gray-500" />;
  }
};

interface PlanStepItemProps {
  step: PlanStep;
  index: number;
  isLast: boolean;
}

function PlanStepItem({ step, index, isLast }: PlanStepItemProps) {
  return (
    <div className="flex">
      <div className="flex flex-col items-center mr-3">
        {getStepIcon(step.status)}
        {!isLast && (
          <div className={`w-0.5 flex-1 mt-1 ${
            step.status === 'completed' ? 'bg-green-400' : 'bg-gray-700'
          }`} />
        )}
      </div>
      <div className={`flex-1 pb-4 ${isLast ? '' : ''}`}>
        <div className={`text-sm ${
          step.status === 'completed' ? 'text-gray-400 line-through' :
          step.status === 'in_progress' ? 'text-white font-medium' :
          step.status === 'skipped' ? 'text-gray-600 line-through' :
          'text-gray-300'
        }`}>
          <span className="text-gray-500 mr-2">{index + 1}.</span>
          {step.description}
        </div>
        {step.substeps && step.substeps.length > 0 && (
          <div className="ml-4 mt-2 space-y-2">
            {step.substeps.map((substep) => (
              <div key={substep.id} className="flex items-center gap-2">
                {getStepIcon(substep.status)}
                <span className={`text-xs ${
                  substep.status === 'completed' ? 'text-gray-500 line-through' :
                  substep.status === 'in_progress' ? 'text-gray-200' :
                  'text-gray-400'
                }`}>
                  {substep.description}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function PlanProgress({ plan }: PlanProgressProps) {
  if (!plan) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 p-4">
        <ListTodo size={32} className="mb-2 opacity-50" />
        <span className="text-sm">No plan created yet</span>
      </div>
    );
  }

  const progressPercent = plan.totalSteps > 0 
    ? Math.round((plan.completedSteps / plan.totalSteps) * 100) 
    : 0;

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2 mb-2">
          <ListTodo size={16} className="text-blue-400" />
          <span className="text-sm font-medium text-gray-200">{plan.title}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <span className="text-xs text-gray-400">
            {plan.completedSteps}/{plan.totalSteps}
          </span>
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Strategy: {plan.strategy}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {plan.steps.map((step, index) => (
          <PlanStepItem 
            key={step.id} 
            step={step} 
            index={index}
            isLast={index === plan.steps.length - 1}
          />
        ))}
      </div>
    </div>
  );
}
