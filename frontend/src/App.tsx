import * as React from "react";
import { ShellController, type ShellState } from "./polling";

type LazyViewsModule = typeof import("./views/LazyViews");
let lazyViewsPromise: Promise<LazyViewsModule> | null = null;

function loadLazyViews(): Promise<LazyViewsModule> {
  if (lazyViewsPromise === null) {
    lazyViewsPromise = import("./views/LazyViews");
  }
  return lazyViewsPromise;
}

const StatusView = React.lazy(() =>
  loadLazyViews().then((module) => ({ default: module.StatusView })),
);
const JobsView = React.lazy(() =>
  loadLazyViews().then((module) => ({ default: module.JobsView })),
);

function currentView(): "status" | "jobs" {
  return window.location.hash === "#jobs" ? "jobs" : "status";
}

function useController(controller: ShellController): ShellState {
  const [state, setState] = React.useState(() => controller.getState());
  React.useEffect(() => {
    const unsubscribe = controller.subscribe(() => setState(controller.getState()));
    controller.start();
    setState(controller.getState());
    return () => {
      unsubscribe();
      controller.dispose();
    };
  }, [controller]);
  return state;
}

export interface AppProps {
  controller?: ShellController;
}

export function App({ controller: suppliedController }: AppProps = {}) {
  const controllerRef = React.useRef<ShellController | null>(null);
  if (controllerRef.current === null) {
    controllerRef.current = suppliedController ?? new ShellController();
  }
  const controller = controllerRef.current;
  const state = useController(controller);
  const [view, setView] = React.useState(currentView);

  React.useEffect(() => {
    const applyHashView = () => setView(currentView());
    window.addEventListener("hashchange", applyHashView);
    return () => window.removeEventListener("hashchange", applyHashView);
  }, []);

  const showJobs = view === "jobs";
  return (
    <>
      <header>
        <div className="header-row">
          <div className="header-row">
            <h1 className="product">Reconstruction Zone</h1>
            <span className="badge">LOCAL ONLY</span>
          </div>
          <div className="muted">
            Last successful refresh: <span id="last-refresh">{state.lastRefresh}</span>
          </div>
        </div>
        <nav aria-label="Local views">
          <a id="status-link" href="#status" aria-current={showJobs ? "false" : "page"}>
            Runtime status
          </a>
          <a id="jobs-link" href="#jobs" aria-current={showJobs ? "page" : "false"}>
            Representative jobs
          </a>
        </nav>
      </header>
      <main>
        <React.Suspense fallback={null}>
          <StatusView state={state} hidden={showJobs} />
          <JobsView state={state} hidden={!showJobs} onSelectJob={(jobId) => controller.selectJob(jobId)} />
        </React.Suspense>
        <p id="live-feedback" className="muted" aria-live="polite">
          {state.liveFeedback}
        </p>
      </main>
    </>
  );
}
