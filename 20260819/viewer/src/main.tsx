import React from "react";
import ReactDOM from "react-dom/client";
import { ChevronLeft, ChevronRight, Download, RefreshCw, RotateCcw, RotateCw } from "lucide-react";
import "./styles.css";

type PageImage = {
  page: number;
  width: number;
  height: number;
  image_url: string;
};

type LayoutRecord = {
  id: string;
  engine: string;
  page: number;
  seq_no: number;
  bbox: [number, number, number, number] | number[];
  category: string;
  text: string;
  confidence: number | null;
  raw_type: string;
};

type EngineStatus = {
  engine: string;
  label: string;
  available: boolean;
  message: string;
  count: number;
  elapsed_seconds?: number | null;
};

type ViewerData = {
  run_id: string;
  pdf_name: string;
  pages: PageImage[];
  source_page_count?: number;
  records: LayoutRecord[];
  engines: EngineStatus[];
};

function getRunId(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get("run_id");
}

function trimText(value: string, maxLength = 5000): string {
  const normalized = (value || "").replace(/\r\n/g, "\n").trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 1)}…`;
}

function notifySelectedPage(page: string, runId: string | null) {
  const normalizedPage = page.trim();
  if (!/^[1-9]\d*$/.test(normalizedPage) || window.parent === window) return;

  window.parent.postMessage(
    {
      type: "pdf-layout-lab:selected-page",
      run_id: runId,
      page: normalizedPage,
    },
    window.location.origin,
  );
}

function normalizeRotation(rotation: number): number {
  const normalized = rotation % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

function formatEngineLabel(engine: EngineStatus): string {
  const state = engine.available ? "有効" : "無効";
  return `${engine.label}: ${state} / ${engine.count} 件`;
}

function App() {
  const runId = getRunId();
  const [data, setData] = React.useState<ViewerData | null>(null);
  const [error, setError] = React.useState<string>("");
  const [loading, setLoading] = React.useState<boolean>(true);
  const [pageNumber, setPageNumber] = React.useState<number>(1);
  const [selectedEngine, setSelectedEngine] = React.useState<string>("");
  const [selectedId, setSelectedId] = React.useState<string>("");
  const [goToPage, setGoToPage] = React.useState<string>("");
  const [rotations, setRotations] = React.useState<Record<number, number>>({});
  const tableRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!runId) {
      setError("run_id が指定されていません。Gradio 画面から解析を実行してください。");
      setLoading(false);
      return;
    }
    fetch(`/artifacts/${runId}/viewer-data.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`viewer-data.json を読み込めませんでした (${response.status})`);
        return response.json();
      })
      .then((payload: ViewerData) => {
        const firstPage = payload.pages[0]?.page ?? 1;
        const firstEngine =
          payload.engines.find((engine) => engine.available && engine.count > 0)?.engine ??
          payload.engines.find((engine) => engine.available)?.engine ??
          payload.engines[0]?.engine ??
          "";
        setData(payload);
        setRotations({});
        setPageNumber(firstPage);
        setSelectedEngine(firstEngine);
        setGoToPage(String(firstPage));
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const sourcePageCount = data?.source_page_count ?? data?.pages[data.pages.length - 1]?.page ?? 1;
  const page = data?.pages.find((item) => item.page === pageNumber) ?? null;
  const selectedRecord = React.useMemo(
    () => data?.records.find((record) => record.id === selectedId) ?? null,
    [data, selectedId],
  );
  const currentEngine = React.useMemo(
    () => data?.engines.find((engine) => engine.engine === selectedEngine) ?? null,
    [data, selectedEngine],
  );
  const pageRecords = React.useMemo(() => {
    if (!data) return [];
    return data.records
      .filter((record) => record.engine === selectedEngine && record.page === pageNumber)
      .sort((a, b) => a.seq_no - b.seq_no);
  }, [data, pageNumber, selectedEngine]);

  React.useEffect(() => {
    if (!data) return;
    const selectedStillVisible = data.records.some(
      (record) => record.id === selectedId && record.engine === selectedEngine && record.page === pageNumber,
    );
    if (!selectedStillVisible) setSelectedId("");
  }, [data, pageNumber, selectedEngine, selectedId]);

  React.useEffect(() => {
    notifySelectedPage(goToPage || String(pageNumber), runId);
  }, [goToPage, pageNumber, runId]);

  function changePage(nextPage: number) {
    if (!data || nextPage < 1 || nextPage > sourcePageCount) return;
    setPageNumber(nextPage);
    setGoToPage(String(nextPage));
  }

  function rotatePage(delta: number) {
    setRotations((prev) => ({ ...prev, [pageNumber]: normalizeRotation((prev[pageNumber] ?? 0) + delta) }));
  }

  function changeEngine(engineId: string) {
    setSelectedEngine(engineId);
    setSelectedId("");
  }

  function selectRecord(record: LayoutRecord) {
    setSelectedEngine(record.engine);
    setSelectedId(record.id);
    if (record.page !== pageNumber) changePage(record.page);
  }

  function inputPage(value: string) {
    setGoToPage(value);
    const page = Number(value);
    if (Number.isInteger(page) && page >= 1 && page <= sourcePageCount) setPageNumber(page);
  }

  if (loading) {
    return (
      <main className="center-state">
        <RefreshCw aria-hidden="true" className="spin" />
        <p>ビューアデータを読み込んでいます。</p>
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="center-state error-state">
        <h1>表示できません</h1>
        <p>{error || "解析結果がありません。"}</p>
      </main>
    );
  }

  const isPreviewOnly = data.engines.length === 0;
  const rotation = normalizeRotation(rotations[pageNumber] ?? 0);

  return (
    <main className="jsonl-shell">
      <header className="jsonl-header">
        <div>
          <p>Run ID: {data.run_id}</p>
          <h1>{data.pdf_name}</h1>
        </div>
        {!isPreviewOnly && (
          <nav className="download-actions" aria-label="ダウンロード">
            <a href={`/artifacts/${data.run_id}/results.json`} download>
              <Download aria-hidden="true" />
              JSON
            </a>
            <a href={`/artifacts/${data.run_id}/results.jsonl`} download>
              <Download aria-hidden="true" />
              JSONL
            </a>
          </nav>
        )}
      </header>

      <section className="jsonl-controls" aria-label="ページ操作">
        <button
          type="button"
          aria-label="前のページ"
          disabled={pageNumber <= 1}
          onClick={() => changePage(pageNumber - 1)}
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <div className="page-jump">
          <input
            aria-label="ページ番号"
            type="number"
            min={1}
            max={sourcePageCount}
            value={goToPage}
            onChange={(event) => inputPage(event.target.value)}
          />
          <span>/ {sourcePageCount}</span>
        </div>
        <div className="page-tools">
          <button type="button" aria-label="左に回転" title="左に回転" onClick={() => rotatePage(-90)}>
            <RotateCcw aria-hidden="true" />
          </button>
          <button type="button" aria-label="右に回転" title="右に回転" onClick={() => rotatePage(90)}>
            <RotateCw aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          aria-label="次のページ"
          disabled={pageNumber >= sourcePageCount}
          onClick={() => changePage(pageNumber + 1)}
        >
          <ChevronRight aria-hidden="true" />
        </button>
      </section>

      {isPreviewOnly ? (
        <p className="engine-message">
          解析前プレビューです。アップロードしたファイルの全 {data.pages.length} ページを表示できます。
        </p>
      ) : (
        <section className="engine-strip" aria-label="解析エンジン">
          {data.engines.map((engine) => (
            <button
              key={engine.engine}
              type="button"
              className={engine.engine === selectedEngine ? "active-engine" : ""}
              onClick={() => changeEngine(engine.engine)}
              title={engine.message}
            >
              <span>{formatEngineLabel(engine)}</span>
            </button>
          ))}
        </section>
      )}

      {!isPreviewOnly && currentEngine && (
        <p className={currentEngine.available ? "engine-message" : "engine-message warning-message"}>
          {currentEngine.label}: {currentEngine.message}
        </p>
      )}

      <section
        className={isPreviewOnly ? "split-viewer preview-only" : "split-viewer"}
        aria-label={isPreviewOnly ? "ファイルプレビュー" : "ファイルと JSONL"}
      >
        <PDFPane
          page={page ?? { page: pageNumber, width: 0, height: 0, image_url: `/page-image/${data.run_id}/${pageNumber}.png` }}
          rotation={rotation}
          note={
            page
              ? undefined
              : `${pageNumber} ページ目はこの解析に含まれていません。プレビュー表示です。このまま「解析を実行」すると、このページを解析します。`
          }
          selectedRecord={selectedRecord}
          onClear={() => setSelectedId("")}
        />
        {!isPreviewOnly && (
          <JSONLTable
            records={pageRecords}
            selectedId={selectedId}
            onSelect={selectRecord}
            scrollRef={tableRef}
          />
        )}
      </section>
    </main>
  );
}

function PDFPane(props: {
  page: PageImage;
  rotation: number;
  note?: string;
  selectedRecord: LayoutRecord | null;
  onClear: () => void;
}) {
  const { page, rotation, note, selectedRecord, onClear } = props;
  const paneRef = React.useRef<HTMLDivElement>(null);
  const highlightRef = React.useRef<HTMLDivElement>(null);
  // 未解析ページのプレビューは寸法を持たないので、画像の読み込み時に実寸を測る
  const [measured, setMeasured] = React.useState<{ width: number; height: number } | null>(null);
  const width = page.width || measured?.width || 0;
  const height = page.height || measured?.height || 0;
  const aspectRatio = width && height ? width / height : 1;
  const isQuarterTurn = rotation % 180 !== 0;
  const highlight =
    selectedRecord && selectedRecord.page === page.page && page.width ? getImageBoxStyle(selectedRecord, page) : null;

  React.useEffect(() => {
    if (!selectedRecord || selectedRecord.page !== page.page || !paneRef.current) return;
    const frameId = window.requestAnimationFrame(() => {
      const pane = paneRef.current;
      const highlightElement = highlightRef.current;
      if (!pane || !highlightElement) return;

      scrollPaneToElement(pane, highlightElement);
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [page.page, selectedRecord]);

  return (
    <div className="pdf-pane" ref={paneRef} onClick={onClear}>
      {note && <p className="preview-note">{note}</p>}
      <div
        className="pdf-page-frame"
        style={{
          width: width ? `${width}px` : "100%",
          maxWidth: "100%",
          aspectRatio: `${isQuarterTurn ? 1 / aspectRatio : aspectRatio}`,
        }}
      >
        <div
          className="pdf-page-wrap"
          style={{
            width: isQuarterTurn ? `${aspectRatio * 100}%` : "100%",
            height: isQuarterTurn ? `${(1 / aspectRatio) * 100}%` : "100%",
            transform: `translate(-50%, -50%) rotate(${rotation}deg)`,
          }}
        >
          <img
            src={page.image_url}
            alt={`${page.page} ページ目`}
            draggable={false}
            onLoad={(event) => {
              if (page.width) return;
              const image = event.currentTarget;
              setMeasured({ width: image.naturalWidth, height: image.naturalHeight });
            }}
          />
          {highlight && (
            <div
              ref={highlightRef}
              className="selected-highlight"
              style={highlight}
              title={`${selectedRecord?.seq_no ?? ""} ${selectedRecord?.category ?? ""}`}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function JSONLTable(props: {
  records: LayoutRecord[];
  selectedId: string;
  onSelect: (record: LayoutRecord) => void;
  scrollRef: React.RefObject<HTMLDivElement>;
}) {
  const { records, selectedId, onSelect, scrollRef } = props;

  return (
    <div className="jsonl-table-pane" ref={scrollRef}>
      <table className="jsonl-table">
        <thead>
          <tr>
            <th>Page<br />No</th>
            <th>Seq<br />No</th>
            <th>Sentence</th>
            <th>Detected<br />Type</th>
          </tr>
        </thead>
        <tbody>
          {records.length === 0 ? (
            <tr>
              <td colSpan={4} className="empty-cell">
                このページに表示できる JSONL 行がありません。
              </td>
            </tr>
          ) : (
            records.map((record) => (
              <tr
                key={record.id}
                className={record.id === selectedId ? "jsonl-row-selected" : ""}
                onClick={() => onSelect(record)}
              >
                <td>{record.page}</td>
                <td>{record.seq_no}</td>
                <td className="sentence-cell">{trimText(record.text || record.raw_type || "")}</td>
                <td>{record.category}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function scrollPaneToElement(pane: HTMLElement, element: HTMLElement) {
  const paneRect = pane.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();
  const currentTop = pane.scrollTop;
  const elementTop = currentTop + elementRect.top - paneRect.top;
  const centeredTop = elementTop - (pane.clientHeight - elementRect.height) / 2;
  const maxTop = pane.scrollHeight - pane.clientHeight;

  pane.scrollTo({
    top: Math.max(0, Math.min(centeredTop, maxTop)),
    behavior: "smooth",
  });
}

function getImageBoxStyle(record: LayoutRecord, page: PageImage): React.CSSProperties {
  const [x1, y1, x2, y2] = record.bbox;
  return {
    left: `${(x1 / page.width) * 100}%`,
    top: `${(y1 / page.height) * 100}%`,
    width: `${((x2 - x1) / page.width) * 100}%`,
    height: `${((y2 - y1) / page.height) * 100}%`,
  };
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
