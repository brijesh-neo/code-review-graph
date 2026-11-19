/**
 * CodeLens above each function and class: how many callers it has, how many
 * of those have no test, and how many tests and flows reach it.
 *
 * Reads the graph through the same SqliteReader as the tree views, so it
 * costs a few indexed queries per open file. Turn it off with
 * `codeReviewGraph.codeLens.enabled`.
 */

import * as vscode from "vscode";

import type { GraphNode, LensSummary, SqliteReader } from "../backend/sqlite";

export const SHOW_CALLERS_COMMAND = "codeReviewGraph.codeLens.showCallers";

export function lensTitle(s: LensSummary): string {
  const parts = [`${s.callers.length} caller${s.callers.length === 1 ? "" : "s"}`];
  if (s.untestedCallers > 0) { parts.push(`${s.untestedCallers} untested`); }
  parts.push(`${s.tests} test${s.tests === 1 ? "" : "s"}`);
  if (s.flows > 0) { parts.push(`${s.flows} flow${s.flows === 1 ? "" : "s"}`); }
  return parts.join(" · ");
}

export class GraphCodeLensProvider implements vscode.CodeLensProvider {
  private readonly changed = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.changed.event;

  constructor(private readonly reader: () => SqliteReader | undefined) {}

  refresh(): void {
    this.changed.fire();
  }

  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    const enabled = vscode.workspace
      .getConfiguration("codeReviewGraph")
      .get<boolean>("codeLens.enabled", true);
    const reader = this.reader();
    if (!enabled || !reader || document.uri.scheme !== "file") { return []; }
    let summaries: LensSummary[];
    try {
      summaries = reader.getLensSummaries(document.uri.fsPath);
    } catch {
      return []; // a graph mid-rebuild; the next refresh draws them
    }
    return summaries
      .filter((s) => s.node.lineStart !== null)
      .map((s) => {
        const line = Math.max(0, (s.node.lineStart ?? 1) - 1);
        const range = new vscode.Range(line, 0, line, 0);
        return new vscode.CodeLens(range, {
          title: lensTitle(s),
          command: s.callers.length > 0 ? SHOW_CALLERS_COMMAND : "",
          arguments: [s.callers],
        });
      });
  }
}

export async function showCallers(callers: GraphNode[]): Promise<void> {
  const pick = await vscode.window.showQuickPick(
    callers.map((c) => ({
      label: c.name,
      description: `${vscode.workspace.asRelativePath(c.filePath)}:${c.lineStart ?? 1}`,
      node: c,
    })),
    { placeHolder: "Callers" },
  );
  if (!pick) { return; }
  const doc = await vscode.workspace.openTextDocument(pick.node.filePath);
  const line = Math.max(0, (pick.node.lineStart ?? 1) - 1);
  await vscode.window.showTextDocument(doc, { selection: new vscode.Range(line, 0, line, 0) });
}
