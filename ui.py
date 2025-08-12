# ui.py
import os
import time
import uuid
import requests

import streamlit as st

# Local constants
DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Minimal RAG UI", page_icon="🧪", layout="wide")
st.title("🧪 Mini RAG Admin Panel")

# -------------------------------
# Sidebar: Config + Ingestion
# -------------------------------
with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", value=st.session_state.get("api_base", DEFAULT_API))
    st.session_state.api_base = api_base.rstrip("/")

    # Auto-generate a unique hidden Store ID for this session to avoid Chroma caching issues
    current_store = st.session_state.get("store_id", "")
    if not current_store:
        st.session_state.store_id = f"store_{uuid.uuid4().hex[:8]}"
        current_store = st.session_state.store_id

    # Check if current store exists (per-run) to drive UI state immediately
    def _store_exists(store: str) -> bool:
        try:
            r = requests.get(f"{st.session_state.api_base}/stores/{store}", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    store_id = current_store
    store_exists = _store_exists(store_id)
    # Warning removed per request

    st.divider()
    st.header("Ingestion")
    # Flash success (shown above any warnings)
    if st.session_state.get("build_success_msg"):
        st.success(st.session_state.get("build_success_msg"))
        # Clear after showing once
        st.session_state.build_success_msg = ""
    if st.session_state.get("delete_success_msg"):
        st.success(st.session_state.get("delete_success_msg"))
        st.session_state.delete_success_msg = ""
    # File uploader resets after successful upload
    if "upload_nonce" not in st.session_state:
        st.session_state.upload_nonce = 0
    up = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
        disabled=store_exists or (not store_id),
        key=f"uploader_{st.session_state.upload_nonce}"
    )
    if up:
        with st.spinner("Uploading files…"):
            files = [("files", (f.name, f.getvalue(), "text/plain")) for f in up]
            resp = requests.post(f"{st.session_state.api_base}/stores/{store_id}/documents", files=files, timeout=60)
            if resp.ok:
                saved = resp.json().get("saved", [])
                st.session_state.last_uploaded_files = saved
                st.session_state.store_locked = True
                st.success(f"Uploaded: {', '.join(saved)}")
                st.session_state.upload_nonce += 1
                st.rerun()
            else:
                st.error(f"Upload failed: {resp.status_code} {resp.text}")
    # Show last uploaded files (persist after rerun)
    if st.session_state.get("last_uploaded_files"):
        last_list = st.session_state.last_uploaded_files
        st.caption(f"Last upload: {', '.join(last_list)}")
    # Place info warning under success (if any)
    if store_id and store_exists:
        st.info("Uploading is disabled while a store exists. Delete the store to change its contents.")

    if st.button("📦 Build Vector Store", disabled=(store_exists or (not store_id))):
        # Policy: Require user to delete existing store before building
        try:
            check = requests.get(f"{st.session_state.api_base}/stores/{store_id}", timeout=15)
        except requests.RequestException as e:
            st.error(f"Failed to check store existence: {e}")
            check = None

        if check is not None and check.status_code == 200:
            st.warning("Store already exists. Please click 'Delete Store' first, then build again.")
            st.stop()
        elif check is not None and check.status_code not in (200, 404):
            st.error(f"Unexpected error checking store: {check.status_code} {check.text}")
        else:
            with st.spinner("Embedding chunks and building Chroma…"):
                body = {"store_id": store_id, "rebuild": True}
                resp = requests.post(f"{st.session_state.api_base}/stores", json=body, timeout=180)
                if resp.ok:
                    info = resp.json()
                    st.session_state.build_success_msg = f"Vector store ready. Chunks: {info.get('num_chunks', 0)}"
                    st.rerun()
                else:
                    st.error(f"Build failed: {resp.status_code} {resp.text}")

    k = st.slider("Top‑k chunks", 1, 10, st.session_state.get("k", 7))
    st.session_state.k = k
    st.caption("Tip: Delete the existing store before building a new one. Upload files first, then Build.")

    if st.button("🗑️ Delete Store"):
        with st.spinner("Deleting…"):
            resp = requests.delete(f"{st.session_state.api_base}/stores/{store_id}", timeout=30)
            if resp.ok:
                # Persist a success message across rerun so user sees confirmation
                st.session_state.delete_success_msg = "Store deleted."
                # Clear and unlock after deletion
                st.session_state.store_id = ""
                st.session_state.store_locked = False
                st.rerun()
            else:
                st.error(f"Delete failed: {resp.status_code} {resp.text}")

# -------------------------------
# Session state for chat history
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Layout: Chat | Summaries | Meta
# -------------------------------
col_chat, col_summ, col_meta = st.columns([2, 1.2, 1])
chat_area = col_chat.container()
summ_area = col_summ.container()
meta_area = col_meta.container()

with chat_area:
    st.subheader("Chat")
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

# Render right-side panes from last run (if any)
latest = st.session_state.get("latest", {})
if latest:
    with summ_area:
        st.subheader("Chunk Summaries")

        # Global toggle (default collapsed)
        expand_all = st.checkbox("Expand all summaries", value=False, key="expand_summaries")

        latest = st.session_state.get("latest", {})
        if latest:
            retrieved = latest.get("retrieved", [])
            summaries = latest.get("summaries", [])

            for i, (sum_txt, (chunk_text, meta)) in enumerate(zip(summaries, retrieved)):
                src = meta.get("source", "unknown")
                idx = meta.get("chunk_index", "NA")
                score = meta.get("score", None)

                header = f"#{i} — (chunk index: {idx}, source: `{src}`" + (
                    f", score: {score}" if score is not None else "") + ")"

                with st.expander(header, expanded=expand_all):
                    st.write(sum_txt)
                    # Optional: quick peek button to see raw chunk text too
                    with st.expander("Show raw chunk text", expanded=False):
                        st.code(chunk_text[:1200] + ("…" if len(chunk_text) > 1200 else ""), language="markdown")

            # Show the rewritten search query used for retrieval
            rq = latest.get("rag_metadata", {}).get("rewritten_query")
            if rq:
                st.markdown("**Search query used by retrieval:**")
                st.code(rq)

with meta_area:
    st.subheader("Metadata")
    with st.expander("RAG metadata (collapsible)", expanded=False):
        st.json(latest.get("rag_metadata", {}))
        st.markdown("---")
    st.subheader("Evaluation")
    with st.container(border=True):
        # Controls
        st.caption("Uses built-in fixture: tests/eval_data/vireo_city.txt")
        eval_k = st.number_input("k (top-k)", min_value=1, max_value=10, value=int(st.session_state.get("eval_k", 4)), step=1, key="eval_k")
        if st.button("Run Retrieval Accuracy", key="btn_run_eval"):
            with st.spinner("Running evaluation… (this may take ~1-2 minutes) "):
                payload = {
                    "k": int(eval_k),
                }
                try:
                    t0 = time.perf_counter()
                    r = requests.post(f"{st.session_state.api_base}/tests/retrieval-accuracy", json=payload, timeout=600)
                    client_ms = int((time.perf_counter() - t0) * 1000)
                    if r.ok:
                        res = r.json()
                        # Prefer server-reported processing time if available
                        server_ms_str = r.headers.get("X-Response-Time-ms")
                        try:
                            server_ms = float(server_ms_str) if server_ms_str is not None else None
                        except Exception:
                            server_ms = None
                        st.session_state.eval_result = {
                            "avg": res.get("average_recall", 0.0),
                            "std": res.get("std_recall", 0.0),
                            "total": res.get("total_questions", 0),
                            "ok": res.get("successful_evaluations", 0),
                            "avg_latency_ms": res.get("avg_latency_ms"),
                            "std_latency_ms": res.get("std_latency_ms"),
                            "latency_ms": server_ms if server_ms is not None else client_ms,
                            "latency_source": "server" if server_ms is not None else "client",
                        }
                        st.success("Evaluation completed")
                    else:
                        st.session_state.eval_result = {"error": f"{r.status_code} {r.text}"}
                        st.error(f"Eval failed: {r.status_code} {r.text}")
                except requests.RequestException as e:
                    st.session_state.eval_result = {"error": str(e)}
                    st.error(f"Request failed: {e}")

        # Display last result
        ev = st.session_state.get("eval_result")
        if ev:
            if "error" in ev:
                st.error(ev["error"])
            else:
                st.metric("Mean Recall", f"{ev['avg']:.3f}")
                st.metric("Std Dev", f"{ev['std']:.3f}")
                if ev.get("avg_latency_ms") is not None:
                    st.metric("Mean Latency (s)", f"{(ev['avg_latency_ms']/1000.0):.3f}")
                    st.metric("Std Latency (s)", f"{(ev.get('std_latency_ms', 0.0)/1000.0):.3f}")
                elif ev.get("latency_ms") is not None:
                    src = ev.get("latency_source", "client")
                    st.caption(f"Evaluation latency ({src}): {ev['latency_ms']:.0f} ms")
                st.caption(f"Questions evaluated: {ev['ok']}/{ev['total']}")

# 🔻 Place the chat input at the very bottom of the page (outside columns)
question = st.chat_input("Ask a question about your documents…")
if question:
    # Append immediately and rerun so the user's message shows up right away
    st.session_state.messages.append(("user", question))
    st.session_state.to_send = question
    st.rerun()

# If there is a pending question from the previous run, process it now
pending = st.session_state.get("to_send")
if pending:
    body = {
        "store_id": st.session_state.store_id,
        "question": pending,
        "k": st.session_state.k,
        "history": st.session_state.messages,
        "include_raw_chunks": True,
    }
    try:
        resp = requests.post(f"{st.session_state.api_base}/query", json=body, timeout=120)
        if not resp.ok:
            st.session_state.messages.append(("assistant", f"Error: {resp.status_code} {resp.text}"))
            st.session_state.to_send = None
            st.rerun()
        data = resp.json()
        answer = data.get("answer", "")
        retrieved_chunks = data.get("retrieved_chunks", [])
        raw_chunks_payload = data.get("raw_chunks", []) or []
        raw_chunks = [(rc.get("text", ""), rc.get("metadata", {})) for rc in raw_chunks_payload]
        metadata = data.get("metadata", {})

        st.session_state.messages.append(("assistant", answer))
        summaries = [chunk.get("summary", "") for chunk in retrieved_chunks]
        st.session_state.latest = {
            "retrieved": raw_chunks,
            "summaries": summaries,
            "rag_metadata": metadata,
        }
        st.session_state.to_send = None
        st.rerun()
    except requests.RequestException as e:
        st.session_state.messages.append(("assistant", f"Request failed: {e}"))
        st.session_state.to_send = None
        st.rerun()
