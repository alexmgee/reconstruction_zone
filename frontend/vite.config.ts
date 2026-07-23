import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { createLogger, defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const logger = createLogger();

logger.warn = (message) => {
  throw new Error(`Vite build warning: ${message}`);
};
logger.warnOnce = logger.warn;

function removeDeadInlinedPreloadHelper(): Plugin {
  return {
    name: "remove-dead-inlined-preload-helper",
    apply: "build",
    enforce: "post",
    generateBundle(_options, bundle) {
      let transformedBoundaries = 0;
      const chunks = Object.values(bundle).filter((output) => output.type === "chunk");
      if (chunks.length !== 1) {
        this.error(`expected exactly one JavaScript chunk before inlining; found ${chunks.length}`);
      }

      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        const externalDynamicImports = output.dynamicImports.filter((fileName) => fileName !== output.fileName);
        if (externalDynamicImports.length !== 0) {
          this.error(`dynamic chunks were not inlined: ${externalDynamicImports.join(", ")}`);
        }

        const helper = output.code.indexOf("var __vitePreload = function preload(");
        if (helper === -1) continue;

        const helperPreamble = output.code.lastIndexOf("var scriptRel =", helper);
        const helperEnd = output.code.indexOf("\n//#endregion", helper);
        const callSuffix = ", __VITE_PRELOAD__, import.meta.url)";
        const boundaryCount = output.code.split(callSuffix).length - 1;
        if (helperPreamble === -1 || helperEnd === -1 || boundaryCount !== 1) {
          const boundary = output.code.indexOf("__vitePreload(", helper);
          this.error(
            `unexpected Vite preload-helper shape for the single lazy boundary ` +
              `(preamble=${helperPreamble}, end=${helperEnd}, boundaries=${boundaryCount}, ` +
              `call=${JSON.stringify(output.code.slice(boundary, boundary + 180))})`,
          );
        }

        output.code =
          output.code.slice(0, helperPreamble) +
          "var __vitePreload = function preload(baseModule) {\n\treturn baseModule();\n};" +
          output.code.slice(helperEnd);
        output.code = output.code.replace(callSuffix, ")");
        transformedBoundaries += boundaryCount;
      }

      if (transformedBoundaries !== 1) {
        this.error(`expected exactly one inlined lazy boundary; found ${transformedBoundaries}`);
      }
    },
  };
}

export default defineConfig({
  envDir: false,
  customLogger: logger,
  plugins: [
    react(),
    removeDeadInlinedPreloadHelper(),
    viteSingleFile({
      removeViteModuleLoader: true,
    }),
  ],
  build: {
    minify: false,
    sourcemap: false,
    cssCodeSplit: false,
    emptyOutDir: true,
    modulePreload: false,
    rolldownOptions: {
      output: {
        codeSplitting: false,
      },
    },
  },
});
