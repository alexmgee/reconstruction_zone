import * as React from "react";

const Placeholder = React.lazy(() => import("./views/Placeholder"));

export function App() {
  return (
    <main>
      <h1>Reconstruction Zone</h1>
      <React.Suspense fallback={<p>Loading proof boundary…</p>}>
        <Placeholder />
      </React.Suspense>
    </main>
  );
}
