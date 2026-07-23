import type { ShellState, RegionStatus } from "../polling";
import { LOG_CLASSES, STATE_CLASSES } from "../constants";

interface FieldProps {
  label: string;
  value: string;
}

function Field({ label, value }: FieldProps) {
  return (
    <div className="field">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function booleanText(value: boolean): string {
  return value ? "Yes" : "No";
}

function optionalText(value: string | null): string {
  return value === null ? "Not available" : value;
}

export function StatusLine({ id, value, live }: { id: string; value: RegionStatus; live: boolean }) {
  const className = `status-line${value.tone ? ` ${value.tone}` : ""}`;
  if (live) {
    return (
      <p id={id} className={className} aria-live="polite">
        {value.text}
      </p>
    );
  }
  return (
    <p id={id} className={className}>
      {value.text}
    </p>
  );
}

export interface ViewProps {
  state: ShellState;
  hidden: boolean;
}

export function StatusView({ state, hidden }: ViewProps) {
  const health = state.healthData;
  const version = state.versionData;
  return (
    <section id="status-view" aria-labelledby="status-heading" hidden={hidden}>
      <h2 id="status-heading">Runtime status</h2>
      <StatusLine id="health-status" value={state.healthStatus} live={true} />
      <div className="cards">
        <section className="card" aria-labelledby="health-heading">
          <h3 id="health-heading">Local service</h3>
          <dl id="health-fields" aria-busy={state.healthBusy}>
            {health === null ? null : (
              <>
                <Field label="Healthy" value={booleanText(health.ok)} />
                <Field label="Service" value={health.service} />
                <Field label="Mode" value={health.mode} />
                <Field label="Local only" value={booleanText(health.local_only)} />
                <Field label="State configured" value={booleanText(health.state_configured)} />
              </>
            )}
          </dl>
        </section>
        <section className="card" aria-labelledby="version-heading">
          <h3 id="version-heading">Version</h3>
          <StatusLine id="version-status" value={state.versionStatus} live={false} />
          <dl id="version-fields" aria-busy={state.versionBusy}>
            {version === null ? null : (
              <>
                <Field label="Service" value={version.service} />
                <Field label="Version" value={version.version} />
                <Field label="Source" value={version.version_source} />
              </>
            )}
          </dl>
        </section>
      </div>
    </section>
  );
}

export interface JobsViewProps extends ViewProps {
  onSelectJob(jobId: string): void;
}

export function JobsView({ state, hidden, onSelectJob }: JobsViewProps) {
  const detail = state.jobDetailData;
  return (
    <section id="jobs-view" aria-labelledby="jobs-heading" hidden={hidden}>
      <h2 id="jobs-heading">Representative jobs</h2>
      <StatusLine id="jobs-status" value={state.jobsStatus} live={true} />
      <div className="split">
        <section className="card job-list" aria-labelledby="retained-heading">
          <h3 id="retained-heading">Retained jobs</h3>
          <div id="job-list" aria-busy={state.jobsBusy}>
            {state.jobSummaries?.map((job) => (
              <button
                type="button"
                className={`job-button${job.job_id === state.selectedJobId ? " selected" : ""}`}
                onClick={() => onSelectJob(job.job_id)}
                key={job.job_id}
              >
                <span className="job-heading">
                  {job.job_type}
                  <span className={`pill ${STATE_CLASSES[job.state]}`}>{job.state}</span>
                </span>
                <span className="job-id">{job.job_id}</span>
              </button>
            ))}
          </div>
        </section>
        <section
          id="job-detail"
          className="card job-detail"
          aria-labelledby="detail-heading"
          hidden={!state.detailVisible}
        >
          <h3 id="detail-heading">Selected job</h3>
          <StatusLine id="detail-status" value={state.detailStatus} live={false} />
          <progress id="job-progress" max="100" value={detail?.progress ?? 0}>
            {detail?.progress ?? 0}%
          </progress>
          <dl id="detail-fields">
            {detail === null ? null : (
              <>
                <Field label="Job ID" value={detail.job_id} />
                <Field label="Type" value={detail.job_type} />
                <Field label="State" value={detail.state} />
                <Field label="Progress" value={`${detail.progress}%`} />
                <Field label="Current step" value={optionalText(detail.current_step)} />
                <Field label="Created" value={detail.created_at} />
                <Field label="Started" value={optionalText(detail.started_at)} />
                <Field label="Finished" value={optionalText(detail.finished_at)} />
                {detail.result === null ? null : (
                  <>
                    <Field label="Result kind" value={detail.result.kind} />
                    <Field label="Steps completed" value={String(detail.result.steps_completed)} />
                  </>
                )}
                {detail.error === null ? null : (
                  <>
                    <Field label="Error code" value={detail.error.code} />
                    <Field label="Error" value={detail.error.message} />
                  </>
                )}
              </>
            )}
          </dl>
          <h3>Logs</h3>
          <ol id="job-logs" className="logs">
            {detail?.logs.map((log, index) => (
              <li className={LOG_CLASSES[log.level]} key={`${log.time}-${log.code}-${index}`}>
                {log.time} [{log.level}] {log.code}: {log.message}
              </li>
            ))}
          </ol>
        </section>
      </div>
    </section>
  );
}
