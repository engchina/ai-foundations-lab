import React from "react";
import ReactDOM from "react-dom/client";
import { ChevronLeft, ChevronRight, Download, RefreshCw } from "lucide-react";
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
        setPageNumber(firstPage);
        setSelectedEngine(firstEngine);
        setGoToPage(String(firstPage));
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const page = data?.pages.find((item) => item.page === pageNumber) ?? data?.pages[0];
  const selectedRecord = React.useMemo(
    () => data?.records.find((record) => record.id === selectedId) ?? null,
    [data, selectedId],
  );
  const currentEngine = React.useMemo(
    () => data?.engines.find((engine) => engine.engine === selectedEngine) ?? null,
    [data, selectedEngine],
  );
  const pageRecords = React.useMemo(() => {
    if (!data || !page) return [];
    return data.records
      .filter((record) => record.engine === selectedEngine && record.page === page.page)
      .sort((a, b) => a.seq_no - b.seq_no);
  }, [data, page, selectedEngine]);

  React.useEffect(() => {
    if (!data) return;
    const selectedStillVisible = data.records.some(
      (record) => record.id === selectedId && record.engine === selectedEngine && record.page === pageNumber,
    );
    if (!selectedStillVisible) setSelectedId("");
  }, [data, pageNumber, selectedEngine, selectedId]);

  function changePage(nextPage: number) {
    if (!data) return;
    const pages = data.pages.map((item) => item.page);
    if (!pages.includes(nextPage)) return;
    setPageNumber(nextPage);
    setGoToPage(String(nextPage));
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

  function submitGoToPage(event: React.FormEvent) {
    event.preventDefault();
    const value = Number(goToPage);
    if (Number.isInteger(value)) changePage(value);
  }

  if (loading) {
    return (
      <main className="center-state">
        <RefreshCw aria-hidden="true" className="spin" />
        <p>ビューアデータを読み込んでいます。</p>
      </main>
    );
  }

  if (error || !data || !page) {
    return (
      <main className="center-state error-state">
        <h1>表示できません</h1>
        <p>{error || "解析結果がありません。"}</p>
      </main>
    );
  }

  const pageIndex = data.pages.findIndex((item) => item.page === page.page);
  const previousPage = data.pages[Math.max(0, pageIndex - 1)]?.page ?? page.page;
  const nextPage = data.pages[Math.min(data.pages.length - 1, pageIndex + 1)]?.page ?? page.page;

  return (
    <main className="jsonl-shell">
      <header className="jsonl-header">
        <div>
          <p>Run ID: {data.run_id}</p>
          <h1>{data.pdf_name}</h1>
        </div>
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
      </header>

      <section className="jsonl-controls" aria-label="ページ操作">
        <button
          type="button"
          aria-label="前のページ"
          disabled={pageIndex <= 0}
          onClick={() => changePage(previousPage)}
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <form onSubmit={submitGoToPage} className="page-jump">
          <span>
            ページ {page.page} / {data.pages[data.pages.length - 1]?.page ?? page.page}
          </span>
          <input
            aria-label="ページ番号"
            type="number"
            min={data.pages[0]?.page ?? 1}
            max={data.pages[data.pages.length - 1]?.page ?? 1}
            value={goToPage}
            onChange={(event) => setGoToPage(event.target.value)}
          />
          <button type="submit">移動</button>
        </form>
        <button
          type="button"
          aria-label="次のページ"
          disabled={pageIndex >= data.pages.length - 1}
          onClick={() => changePage(nextPage)}
        >
          <ChevronRight aria-hidden="true" />
        </button>
      </section>

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

      {currentEngine && (
        <p className={currentEngine.available ? "engine-message" : "engine-message warning-message"}>
          {currentEngine.label}: {currentEngine.message}
        </p>
      )}

      <section className="split-viewer" aria-label="PDF と JSONL">
        <PDFPane page={page} selectedRecord={selectedRecord} onClear={() => setSelectedId("")} />
        <JSONLTable
          records={pageRecords}
          selectedId={selectedId}
          onSelect={selectRecord}
          scrollRef={tableRef}
        />
      </section>
    </main>
  );
}

function PDFPane(props: { page: PageImage; selectedRecord: LayoutRecord | null; onClear: () => void }) {
  const { page, selectedRecord, onClear } = props;
  const paneRef = React.useRef<HTMLDivElement>(null);
  const highlightRef = React.useRef<HTMLDivElement>(null);
  const highlight =
    selectedRecord && selectedRecord.page === page.page ? getImageBoxStyle(selectedRecord, page) : null;

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
      <div className="pdf-page-wrap" style={{ width: `${page.width}px`, maxWidth: "100%" }}>
        <img src={page.image_url} alt={`${page.page} ページ目`} draggable={false} />
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
