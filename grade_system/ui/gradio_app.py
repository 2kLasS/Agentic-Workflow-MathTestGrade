from __future__ import annotations

import os
from html import escape
from typing import Any

import gradio as gr

from grade_system.ui.api_client import ApiClientError, GradeSystemApiClient

DEFAULT_API_BASE_URL = os.getenv("GRADIO_API_BASE_URL", "http://localhost:8000")
DEFAULT_TASK_STATUS = "等待提交"
DEFAULT_RESULT_TITLE = "批改结果"
DEFAULT_RESULT_BODY = "提交题目后，这里会显示答案判定和错误归因。"

STATUS_LABELS = {
    "PENDING": "排队中",
    "RUNNING": "批改中",
    "SUCCEEDED": "已完成",
    "FAILED": "失败",
}

APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md",
)

APP_CSS = """
:root,
body,
.gradio-container {
    color-scheme: light !important;
    --body-background-fill: #eef4ff !important;
    --body-background-fill-dark: #eef4ff !important;
    --background-fill-primary: #eef4ff !important;
    --background-fill-primary-dark: #eef4ff !important;
    --background-fill-secondary: #f7fbff !important;
    --background-fill-secondary-dark: #f7fbff !important;
    --block-background-fill: #f9fbff !important;
    --block-background-fill-dark: #f9fbff !important;
    --block-border-color: rgba(148, 163, 184, 0.24) !important;
    --block-border-color-dark: rgba(148, 163, 184, 0.24) !important;
    --input-background-fill: #f7fbff !important;
    --input-background-fill-dark: #f7fbff !important;
    --input-border-color: rgba(148, 163, 184, 0.32) !important;
    --input-border-color-dark: rgba(148, 163, 184, 0.32) !important;
    --body-text-color: #0f172a !important;
    --body-text-color-dark: #0f172a !important;
    --body-text-color-subdued: #64748b !important;
    --body-text-color-subdued-dark: #64748b !important;
    --button-primary-background-fill: #dbeafe !important;
    --button-primary-background-fill-hover: #bfdbfe !important;
    --button-primary-border-color: rgba(96, 165, 250, 0.48) !important;
    --button-primary-border-color-hover: rgba(59, 130, 246, 0.58) !important;
    --button-primary-text-color: #0f172a !important;
    --button-primary-background-fill-dark: #dbeafe !important;
    --button-primary-background-fill-hover-dark: #bfdbfe !important;
    --button-primary-border-color-dark: rgba(96, 165, 250, 0.48) !important;
    --button-primary-border-color-hover-dark: rgba(59, 130, 246, 0.58) !important;
    --button-primary-text-color-dark: #0f172a !important;
    --button-secondary-background-fill: #eff6ff !important;
    --button-secondary-background-fill-hover: #dbeafe !important;
    --button-secondary-border-color: rgba(148, 163, 184, 0.28) !important;
    --button-secondary-border-color-hover: rgba(96, 165, 250, 0.42) !important;
    --button-secondary-text-color: #0f172a !important;
    --button-secondary-background-fill-dark: #eff6ff !important;
    --button-secondary-background-fill-hover-dark: #dbeafe !important;
    --button-secondary-border-color-dark: rgba(148, 163, 184, 0.28) !important;
    --button-secondary-border-color-hover-dark: rgba(96, 165, 250, 0.42) !important;
    --button-secondary-text-color-dark: #0f172a !important;
    --gs-bg: #eef4ff;
    --gs-bg-soft: #f7fbff;
    --gs-panel: #fbfdff;
    --gs-panel-alt: #f4f8ff;
    --gs-border: rgba(148, 163, 184, 0.24);
    --gs-workspace-bg: #eff6ff;
    --gs-workspace-panel: #eff6ff;
    --gs-workspace-panel-strong: #dbeafe;
    --gs-workspace-border: rgba(147, 197, 253, 0.72);
    --gs-workspace-border-soft: rgba(147, 197, 253, 0.46);
    --gs-input: #f7fbff;
    --gs-ink: #0f172a;
    --gs-ink-soft: #334155;
    --gs-muted: #64748b;
    --gs-accent: #60a5fa;
    --gs-accent-strong: #3b82f6;
    --gs-danger: #fb7185;
    --gs-success: #34d399;
}

html,
body,
body > .gradio-container,
.gradio-container,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {
    min-height: 100vh;
    background:
        radial-gradient(circle at top, rgba(96, 165, 250, 0.28), transparent 28%),
        linear-gradient(180deg, #eef4ff 0%, #f8fbff 52%, #f2f7ff 100%) !important;
    color: var(--gs-ink) !important;
}

.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.app-shell,
.auth-stage,
.auth-stage > div,
.auth-shell {
    background: transparent !important;
}

.gradio-container,
.gradio-container .prose,
.gradio-container .gr-markdown,
.gradio-container .gr-markdown p,
.gradio-container .gr-markdown span,
.gradio-container .gr-markdown strong,
.gradio-container .gr-markdown li,
.gradio-container .gr-markdown code,
.gradio-container label,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container table,
.gradio-container th,
.gradio-container td {
    color: var(--gs-ink) !important;
}

.app-shell {
    max-width: 1480px;
    margin: 0 auto;
    padding: 28px 24px 40px;
}

.app-header {
    margin-bottom: 20px;
    padding: 4px 2px;
}

.app-header .eyebrow {
    display: inline-block;
    margin-bottom: 10px;
    color: var(--gs-accent);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.app-header h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    line-height: 1.15;
    color: #111111 !important;
}

.app-header p {
    margin: 8px 0 0;
    font-size: 15px;
    color: var(--gs-muted) !important;
}

.banner-text {
    margin: 0 0 16px;
    padding: 12px 14px;
    border-radius: 14px !important;
    border: 1px solid transparent !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
    overflow: visible !important;
    max-height: none !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.banner-text::-webkit-scrollbar,
.banner-text *::-webkit-scrollbar,
.workspace-intro::-webkit-scrollbar,
.workspace-intro *::-webkit-scrollbar {
    display: none !important;
}

.banner-text > div,
.banner-text [data-testid="html"],
.banner-text .html-container,
.banner-text .prose,
.banner-inner {
    overflow: visible !important;
    max-height: none !important;
    padding: 0 !important;
    white-space: normal !important;
}

.user-banner {
    background: #eff6ff !important;
    border-color: rgba(147, 197, 253, 0.72) !important;
    color: #111827 !important;
}

.error-banner {
    background: #fef2f2 !important;
    border-color: rgba(252, 165, 165, 0.72) !important;
    color: #111827 !important;
}

.user-banner p,
.user-banner span,
.user-banner strong,
.user-banner code,
.error-banner p,
.error-banner span,
.error-banner strong,
.error-banner code {
    color: #111827 !important;
}

.user-banner code,
.error-banner code {
    background: #dbeafe !important;
    border: 1px solid rgba(147, 197, 253, 0.65) !important;
}

.auth-stage {
    display: flex;
    justify-content: center;
}

.auth-shell {
    width: min(100%, 440px);
    margin: 44px auto 0;
}

.card-panel {
    background: linear-gradient(180deg, #fbfdff 0%, #f6faff 100%) !important;
    border: 1px solid var(--gs-border) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 32px rgba(148, 163, 184, 0.14) !important;
    padding: 20px !important;
    overflow: hidden;
}

.surface-card {
    min-height: 100%;
    overflow: visible !important;
}

.card-panel h2,
.card-panel h3 {
    margin: 0 0 14px;
    font-weight: 650;
}

.card-panel p {
    color: var(--gs-muted) !important;
}

.auth-card {
    background: #ffffff !important;
    border-color: rgba(226, 232, 240, 0.96) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.98),
        0 18px 40px rgba(15, 23, 42, 0.05) !important;
    padding: 28px !important;
}

.auth-card .gr-group,
.auth-card .gr-block,
.auth-card .gr-box,
.auth-card .gr-panel,
.auth-card .gr-form,
.auth-card .gr-column,
.auth-card .gr-row {
    background: transparent !important;
}

.auth-card [role="tabpanel"],
.auth-card [data-testid="tab-item"],
.auth-card .gradio-tabitem,
.auth-card .gr-tabitem {
    background: #fcfdff !important;
    border: 1px solid rgba(241, 245, 249, 0.98) !important;
    border-radius: 16px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.98) !important;
    color: #111827 !important;
}

.auth-card [role="tabpanel"] > div,
.auth-card [data-testid="tab-item"] > div {
    background: transparent !important;
    padding: 14px !important;
}

.auth-title {
    margin-bottom: 18px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(226, 232, 240, 0.88);
}

.auth-title h2 {
    margin: 0;
    font-size: 24px;
}

.auth-title p {
    margin: 8px 0 0;
    font-size: 14px;
    color: var(--gs-muted) !important;
}

.auth-card [role="tablist"] {
    gap: 8px;
    margin-bottom: 16px;
    padding: 4px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(241, 245, 249, 0.98) !important;
    background: #f8fafc !important;
}

.auth-card button[role="tab"] {
    border-radius: 12px !important;
    border: 1px solid rgba(226, 232, 240, 0.84) !important;
    background: #ffffff !important;
    color: #0f172a !important;
    box-shadow: none !important;
}

.auth-card button[role="tab"][aria-selected="true"] {
    background: #f2f7ff !important;
    border-color: rgba(191, 219, 254, 0.88) !important;
    color: #0f172a !important;
    box-shadow: 0 4px 12px rgba(219, 234, 254, 0.24) !important;
}

.workspace-head {
    align-items: flex-start !important;
    margin-bottom: 18px;
}

.workspace-head h2 {
    margin: 0;
    font-size: 28px;
}

.workspace-head p {
    margin: 8px 0 0;
    color: var(--gs-muted) !important;
}

.workspace-intro,
.workspace-intro > div,
.workspace-intro [data-testid="html"],
.workspace-intro .html-container {
    overflow: visible !important;
    max-height: none !important;
    margin: 0 !important;
    padding: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.workspace-intro h2 {
    margin: 0 !important;
    font-size: 28px !important;
    line-height: 1.2 !important;
    color: #0f172a !important;
}

.workspace-intro p {
    margin: 8px 0 0 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #64748b !important;
}

.section-title,
.section-title > div,
.section-title [data-testid="html"],
.section-title .html-container {
    overflow: visible !important;
    max-height: none !important;
    margin: 0 !important;
    padding: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.section-title::-webkit-scrollbar,
.section-title *::-webkit-scrollbar {
    display: none !important;
}

.section-heading {
    margin: 0 0 14px !important;
    font-size: 18px !important;
    font-weight: 650 !important;
    letter-spacing: -0.01em;
    line-height: 1.3 !important;
    color: #0f172a !important;
}

.result-title-slot,
.result-title-slot > div,
.result-title-slot [data-testid="html"],
.result-title-slot .html-container {
    display: block !important;
    overflow: visible !important;
    max-height: none !important;
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.result-title-slot::-webkit-scrollbar,
.result-title-slot *::-webkit-scrollbar {
    display: none !important;
}

.result-feedback-card {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
    padding: 14px 18px 12px 18px;
    border-radius: 16px 16px 0 0;
    border: 1px solid rgba(96, 165, 250, 0.24);
    border-bottom: none;
    background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.92),
        0 8px 22px rgba(219, 234, 254, 0.28);
    overflow: hidden;
}

.result-title-slot + .info-box,
.result-title-slot + .result-box,
.result-title-slot + .info-box > div,
.result-title-slot + .result-box > div {
    margin-top: -1px !important;
}

.result-feedback-kicker {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
}

.result-feedback-title {
    font-size: 19px;
    font-weight: 700;
    line-height: 1.2;
    color: #0f172a;
}

.result-feedback-note {
    font-size: 13px;
    line-height: 1.55;
    color: #475569;
}

.result-feedback-progress {
    border-color: rgba(96, 165, 250, 0.30);
    background: linear-gradient(180deg, #eff6ff 0%, #e8f1ff 100%);
}

.result-feedback-progress::before {
    background: #3b82f6;
}

.result-feedback-progress .result-feedback-title {
    color: #1d4ed8;
}

.result-feedback-success {
    border-color: rgba(52, 211, 153, 0.34);
    background: linear-gradient(180deg, #ecfdf5 0%, #f0fdf4 100%);
}

.result-feedback-success::before {
    background: #10b981;
}

.result-feedback-success .result-feedback-title {
    color: #166534;
}

.result-feedback-error {
    border-color: rgba(251, 146, 60, 0.30);
    background: linear-gradient(180deg, #fff7ed 0%, #fffaf0 100%);
}

.result-feedback-error::before {
    background: #f97316;
}

.result-feedback-error .result-feedback-title {
    color: #c2410c;
}

.result-feedback-warning {
    border-color: rgba(245, 158, 11, 0.30);
    background: linear-gradient(180deg, #fffbeb 0%, #fff7ed 100%);
}

.result-feedback-warning::before {
    background: #f59e0b;
}

.result-feedback-warning .result-feedback-title {
    color: #b45309;
}

.result-feedback-danger {
    border-color: rgba(244, 63, 94, 0.28);
    background: linear-gradient(180deg, #fff1f2 0%, #fff7ed 100%);
}

.result-feedback-danger::before {
    background: #ef4444;
}

.result-feedback-danger .result-feedback-title {
    color: #b91c1c;
}

.result-feedback-placeholder {
    border-style: dashed;
    border-color: rgba(148, 163, 184, 0.34);
    background: linear-gradient(180deg, #fbfdff 0%, #f6faff 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.result-feedback-placeholder::before {
    background: #94a3b8;
}

.result-feedback-placeholder .result-feedback-title {
    color: #334155;
}

.result-feedback-placeholder .result-feedback-note {
    color: #64748b;
}

.status-slot,
.status-slot > div,
.status-slot [data-testid="html"],
.status-slot .html-container {
    overflow: hidden !important;
    max-height: none !important;
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.status-slot::-webkit-scrollbar,
.status-slot *::-webkit-scrollbar {
    display: none !important;
}

.status-text {
    display: block;
    overflow: hidden;
    color: #0f172a;
}

.status-line {
    display: block;
    line-height: 1.55;
}

.workspace-stage {
    background: var(--gs-workspace-bg) !important;
    border: 1px solid var(--gs-workspace-border) !important;
    box-shadow: 0 14px 36px rgba(191, 219, 254, 0.28) !important;
    border-radius: 24px !important;
    padding: 18px !important;
}

.workspace-stage > div {
    background: transparent !important;
}

.workspace-stage .gr-group,
.workspace-stage .gr-block,
.workspace-stage .gr-box,
.workspace-stage .gr-panel,
.workspace-stage .gr-form,
.workspace-stage .gr-column,
.workspace-stage .gr-row {
    background: transparent !important;
}

.workspace-stage .card-panel {
    background: var(--gs-workspace-panel) !important;
    border: 1px solid var(--gs-workspace-border-soft) !important;
    box-shadow: 0 10px 28px rgba(191, 219, 254, 0.18) !important;
    overflow: visible !important;
}

.workspace-stage .card-panel h2,
.workspace-stage .card-panel h3,
.workspace-stage .card-panel p,
.workspace-stage .card-panel span,
.workspace-stage .workspace-head h2,
.workspace-stage .workspace-head p {
    color: #111827 !important;
}

.workspace-grid {
    align-items: flex-start !important;
    gap: 16px !important;
}

.surface-card h3 {
    margin: 0 0 14px !important;
    font-size: 18px !important;
    font-weight: 650 !important;
    letter-spacing: -0.01em;
    color: #0f172a !important;
}

.workspace-stage .btn-secondary {
    background: var(--gs-workspace-panel-strong) !important;
    border-color: rgba(147, 197, 253, 0.82) !important;
    color: #0f172a !important;
}

.workspace-stage .btn-secondary:hover {
    background: #bfdbfe !important;
}

.workspace-actions {
    gap: 10px;
    justify-content: flex-end;
    align-items: center !important;
}

.workspace-actions > * {
    flex: 0 0 auto;
}

.surface-card .gr-textbox,
.auth-card .gr-textbox,
.surface-card [data-testid="textbox"],
.auth-card [data-testid="textbox"],
.surface-card .gr-textbox > div,
.auth-card .gr-textbox > div,
.surface-card [data-testid="textbox"] > div,
.auth-card [data-testid="textbox"] > div {
    background: linear-gradient(180deg, #fbfdff 0%, #f4f8ff 100%) !important;
    color: #0f172a !important;
    border: 1px solid rgba(191, 219, 254, 0.42) !important;
    border-radius: 14px !important;
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.8) !important;
}

.surface-card textarea,
.surface-card input,
.auth-card textarea,
.auth-card input {
    background: transparent !important;
    color: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    border-radius: 12px !important;
    padding: 13px 15px !important;
    box-shadow: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    resize: none !important;
    line-height: 1.6 !important;
}

.surface-card textarea::placeholder,
.surface-card input::placeholder,
.auth-card textarea::placeholder,
.auth-card input::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

.surface-card textarea:focus,
.surface-card input:focus,
.auth-card textarea:focus,
.auth-card input:focus {
    background: rgba(255, 255, 255, 0.72) !important;
    border-color: rgba(96, 165, 250, 0.56) !important;
    box-shadow: 0 0 0 3px rgba(191, 219, 254, 0.52) !important;
    outline: none !important;
}

.surface-card textarea {
    min-height: 152px !important;
}

.surface-card input,
.auth-card input {
    min-height: 46px !important;
}

.surface-card label span,
.auth-card label span {
    color: var(--gs-ink-soft) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

.workspace-stage .surface-card label span,
.auth-card label span {
    color: #0f172a !important;
}

.auth-card .gr-textbox,
.auth-card [data-testid="textbox"],
.auth-card .gr-textbox > div,
.auth-card [data-testid="textbox"] > div {
    background: #ffffff !important;
    border-color: rgba(226, 232, 240, 0.9) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.98) !important,
        0 4px 12px rgba(241, 245, 249, 0.9) !important;
}

.auth-card input,
.auth-card textarea {
    background: rgba(255, 255, 255, 0.94) !important;
    border-color: rgba(203, 213, 225, 0.58) !important;
}

.auth-card input:focus,
.auth-card textarea:focus {
    background: #ffffff !important;
    border-color: rgba(96, 165, 250, 0.52) !important;
    box-shadow: 0 0 0 3px rgba(219, 234, 254, 0.62) !important;
}

.btn-primary,
.btn-primary > button,
button.btn-primary {
    min-height: 44px !important;
    border: 1px solid rgba(96, 165, 250, 0.36) !important;
    border-radius: 12px !important;
    background: linear-gradient(180deg, #e8f2ff 0%, #d9e9ff 100%) !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 18px rgba(191, 219, 254, 0.45) !important;
    transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
}

.btn-primary:hover,
.btn-primary > button:hover,
button.btn-primary:hover {
    background: linear-gradient(180deg, #dcebff 0%, #cde2ff 100%) !important;
    box-shadow: 0 10px 22px rgba(191, 219, 254, 0.52) !important;
    transform: translateY(-1px);
}

.btn-secondary,
.btn-secondary > button,
button.btn-secondary {
    min-height: 42px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    background: linear-gradient(180deg, #fafdff 0%, #eef6ff 100%) !important;
    color: #0f172a !important;
    box-shadow: 0 4px 12px rgba(219, 234, 254, 0.42) !important;
    transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
}

.btn-secondary:hover,
.btn-secondary > button:hover,
button.btn-secondary:hover {
    background: linear-gradient(180deg, #eef6ff 0%, #e0efff 100%) !important;
    box-shadow: 0 8px 18px rgba(219, 234, 254, 0.48) !important;
    transform: translateY(-1px);
}

.info-box {
    background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%) !important;
    border: 1px solid rgba(191, 219, 254, 0.62) !important;
    border-radius: 16px !important;
    padding: 14px 16px !important;
    color: #0f172a !important;
    line-height: 1.65 !important;
}

.workspace-stage .info-box {
    background: var(--gs-workspace-panel) !important;
    border-color: var(--gs-workspace-border-soft) !important;
    color: #0f172a !important;
}

.info-box p,
.info-box span,
.info-box strong,
.info-box li,
.info-box code,
.result-box p,
.result-box span,
.result-box strong,
.result-box li {
    color: #0f172a !important;
}

.info-box code {
    background: #e0efff !important;
    border: 1px solid rgba(191, 219, 254, 0.7) !important;
}

.status-box {
    margin-bottom: 16px;
    border-left: 4px solid var(--gs-accent) !important;
}

.result-box {
    min-height: 220px;
    margin-top: -1px !important;
    padding-top: 12px !important;
    border-top: none !important;
    border-top-left-radius: 0 !important;
    border-top-right-radius: 0 !important;
    overflow: visible !important;
    box-shadow: 0 10px 24px rgba(219, 234, 254, 0.22) !important;
}

.result-box > div,
.status-box > div {
    overflow: visible !important;
}

.result-box .prose,
.result-box .markdown,
.result-box [data-testid="markdown"],
.result-box [data-testid="markdown"] > div {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.result-box p:first-child,
.result-box ul:first-child,
.result-box ol:first-child {
    margin-top: 0 !important;
}

.history-table {
    background: var(--gs-workspace-panel) !important;
    border: 1px solid var(--gs-workspace-border-soft) !important;
    border-radius: 16px !important;
    overflow: hidden;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
}

.history-table > div,
.history-table table,
.history-table [role="grid"] {
    background: transparent !important;
    color: #0f172a !important;
    table-layout: fixed !important;
    font-size: 13px !important;
}

.history-table th,
.history-table thead tr {
    background: var(--gs-workspace-panel-strong) !important;
    color: #475569 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

.history-table td {
    background: #eff6ff !important;
    color: #0f172a !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.history-table th,
.history-table td {
    border-color: rgba(148, 163, 184, 0.22) !important;
    padding: 10px 12px !important;
    line-height: 1.35 !important;
    vertical-align: middle !important;
}

.history-table tbody tr td {
    transition: background 0.16s ease, box-shadow 0.16s ease, color 0.16s ease !important;
}

.history-table tbody tr:nth-child(odd) td {
    background: #eff6ff !important;
}

.history-table tbody tr:nth-child(even) td {
    background: #eff6ff !important;
}

.history-table td:nth-child(1),
.history-table th:nth-child(1) {
    width: 54px !important;
}

.history-table td:nth-child(2),
.history-table th:nth-child(2) {
    width: 42% !important;
}

.history-table td:nth-child(3),
.history-table th:nth-child(3) {
    width: 82px !important;
}

.history-table td:nth-child(4),
.history-table th:nth-child(4) {
    width: 76px !important;
}

.history-table td:nth-child(5),
.history-table th:nth-child(5) {
    width: 132px !important;
}

.history-table tbody tr:hover td {
    background: #dbeafe !important;
}

.history-table td.cell-selected,
.history-table .cell-selected {
    --ring-color: rgba(147, 197, 253, 0.82) !important;
    background: #eff6ff !important;
    color: #0f172a !important;
}

.history-table tbody tr:hover td.cell-selected,
.history-table tbody tr:hover .cell-selected {
    background: #dbeafe !important;
}

@media (max-width: 900px) {
    .app-shell {
        padding: 20px 14px 28px;
    }

    .auth-shell {
        margin-top: 18px;
    }

    .workspace-actions {
        justify-content: stretch;
    }
}
"""

HERO_HTML = """
<div class="app-header">
  <div class="eyebrow">Intelligent Grading</div>
  <h1>GradeSystem</h1>
  <p>数学解答题批改工作台</p>
</div>
"""

WORKSPACE_INTRO_HTML = """
<div>
  <h2>批改工作台</h2>
  <p>输入题干和学生答案后即可提交批改。</p>
</div>
"""


def get_api_client() -> GradeSystemApiClient:
    return GradeSystemApiClient(DEFAULT_API_BASE_URL)


def empty_session_state() -> dict[str, Any]:
    return {
        "access_token": "",
        "refresh_token": "",
        "user": {},
    }


def is_logged_in(session_state: dict[str, Any] | None) -> bool:
    return bool(session_state and session_state.get("access_token"))


def render_banner_html(text: str) -> str:
    safe_text = escape(str(text or "")).replace("\n", "<br>")
    return f'<div class="banner-inner">{safe_text}</div>'


def render_section_title_html(text: str) -> str:
    title = str(text or "").strip().lstrip("#").strip() or "未命名"
    return f'<div class="section-heading">{escape(title)}</div>'


def render_result_badge_html(text: str) -> str:
    title = str(text or "").strip().lstrip("#").strip()
    if not title or title == DEFAULT_RESULT_TITLE:
        return (
            '<div class="result-feedback-card result-feedback-placeholder">'
            '<div class="result-feedback-kicker">判定结果</div>'
            '<div class="result-feedback-title">等待判定</div>'
            '<div class="result-feedback-note">提交题目后，这里会持续显示本题当前的判定状态与反馈方向。</div>'
            "</div>"
        )

    tone = "progress"
    note = "系统正在整理本题的判定与反馈，请稍候。"

    if title == "答案正确":
        tone = "success"
        note = "本题已判定为正确，可以直接进入下一题。"
    elif title == "答案错误":
        tone = "error"
        note = "已生成详细错因分析，继续查看下方说明即可。"
    elif title == "批改中":
        tone = "progress"
        note = "系统正在执行判定与归因流程，结果会自动刷新。"
    elif title in {"加载失败", "网络错误"}:
        tone = "warning"
        note = "结果暂时没有成功加载，可以稍后重试。"
    elif title in {"批改失败", "提交失败"}:
        tone = "danger"
        note = "本次处理没有完成，请查看下方提示后重试。"

    return (
        f'<div class="result-feedback-card result-feedback-{tone}">'
        f'<div class="result-feedback-kicker">判定结果</div>'
        f'<div class="result-feedback-title">{escape(title)}</div>'
        f'<div class="result-feedback-note">{escape(note)}</div>'
        f"</div>"
    )


def render_status_html(text: str) -> str:
    lines = [escape(line.strip()) for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    html_lines = "".join(f'<div class="status-line">{line}</div>' for line in lines)
    return f'<div class="status-text">{html_lines}</div>'


def build_user_summary(session_state: dict[str, Any] | None) -> str:
    if not is_logged_in(session_state):
        return render_banner_html("当前未登录")

    user = session_state.get("user", {})
    display_name = str(user.get("display_name", "")).strip()
    username = str(user.get("username", "")).strip()
    if display_name and username:
        return (
            f'<div class="banner-inner">已登录：<strong>{escape(display_name)}</strong> '
            f'({escape(username)})</div>'
        )
    if username:
        return render_banner_html(f"已登录：{username}")
    return render_banner_html("已成功登录")


def normalize_time_display(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return text.replace("T", " ").split(" ", 1)[0]


def result_label(item: dict[str, Any]) -> str:
    status = str(item.get("status", "")).upper()
    if status in {"PENDING", "RUNNING"}:
        return "处理中"
    if status == "FAILED":
        return "批改失败"
    is_correct = item.get("is_correct")
    if is_correct is True:
        return "正确"
    if is_correct is False:
        return "错误"
    return "-"


def compact_text(value: Any, max_length: int = 22) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "-"
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3]}..."


def history_rows(items: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for index, item in enumerate(items, start=1):
        rows.append(
            [
                str(index),
                compact_text(item.get("question_excerpt", "")),
                STATUS_LABELS.get(str(item.get("status", "")).upper(), str(item.get("status", ""))),
                result_label(item),
                normalize_time_display(item.get("created_at")),
            ]
        )
    return rows


def render_task_detail(detail: dict[str, Any] | None) -> tuple[str, str, str]:
    if not detail:
        return render_status_html(DEFAULT_TASK_STATUS), render_result_badge_html(""), DEFAULT_RESULT_BODY

    status = str(detail.get("status", "")).upper()
    status_label = STATUS_LABELS.get(status, status or "-")
    task_status_text = render_status_html(f"任务状态：{status_label} | ID：{detail.get('task_id', '-')}")

    if status in {"PENDING", "RUNNING"}:
        return (
            task_status_text,
            render_result_badge_html("批改中"),
            "系统正在执行题目拆分、步骤分析、正确性判定和归因汇总。"
            "由于当前工作流会多次调用模型，通常需要几十秒到数分钟。",
        )

    if status == "FAILED":
        error_message = str(detail.get("error_message", "")).strip() or "系统处理失败，请稍后重试。"
        return task_status_text, render_result_badge_html("批改失败"), error_message

    if detail.get("is_correct") is True:
        return task_status_text, render_result_badge_html("答案正确"), "该学生的作答完全正确。"

    summary = str(detail.get("attribution_summary_text", "")).strip()
    if not summary:
        summary = "未获取到详细错因，请检查题目。"
    return task_status_text, render_result_badge_html("答案错误"), summary


def fetch_history_data(session_state: dict[str, Any]) -> tuple[list[list[str]], list[dict[str, Any]]]:
    if not is_logged_in(session_state):
        return [], []
    data = get_api_client().list_grading_tasks(session_state["access_token"])
    items = data.get("items", [])
    return history_rows(items), items


def login_user(username: str, password: str):
    try:
        response = get_api_client().login(username.strip(), password)
        session_state = {
            "access_token": response["access_token"],
            "refresh_token": response["refresh_token"],
            "user": response["user"],
        }
        rows, items = fetch_history_data(session_state)
        return (
            session_state,
            gr.update(visible=False),
            gr.update(visible=True),
            build_user_summary(session_state),
            gr.update(visible=False),
            rows,
            items,
            render_status_html(DEFAULT_TASK_STATUS),
            render_result_badge_html(""),
            DEFAULT_RESULT_BODY,
            "submit",
            "",
            "",
            gr.update(value="提交批改", interactive=True),
            "",
            "",
        )
    except ApiClientError as exc:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.update(value=render_banner_html(f"登录失败：{exc}"), visible=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "",
        )


def register_user(username: str, display_name: str, password: str):
    try:
        response = get_api_client().register(
            username=username.strip(),
            password=password,
            display_name=display_name.strip(),
        )
        session_state = {
            "access_token": response["access_token"],
            "refresh_token": response["refresh_token"],
            "user": response["user"],
        }
        rows, items = fetch_history_data(session_state)
        return (
            session_state,
            gr.update(visible=False),
            gr.update(visible=True),
            build_user_summary(session_state),
            gr.update(visible=False),
            rows,
            items,
            render_status_html(DEFAULT_TASK_STATUS),
            render_result_badge_html(""),
            DEFAULT_RESULT_BODY,
            "submit",
            "",
            "",
            gr.update(value="提交批改", interactive=True),
            "",
            "",
            "",
        )
    except ApiClientError as exc:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.update(value=render_banner_html(f"注册失败：{exc}"), visible=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "",
            gr.skip(),
            "",
        )


def logout_user(session_state: dict[str, Any]):
    if is_logged_in(session_state):
        refresh_token = str(session_state.get("refresh_token", "")).strip()
        if refresh_token:
            try:
                get_api_client().logout(
                    refresh_token=refresh_token,
                    access_token=str(session_state.get("access_token", "")),
                )
            except ApiClientError:
                pass

    return (
        empty_session_state(),
        gr.update(visible=True),
        gr.update(visible=False),
        build_user_summary(None),
        gr.update(visible=False),
        [],
        [],
        render_status_html(DEFAULT_TASK_STATUS),
        render_result_badge_html(""),
        DEFAULT_RESULT_BODY,
        "submit",
        "",
        "",
        gr.update(value="提交批改", interactive=True),
        "",
        "",
    )


def refresh_history(session_state: dict[str, Any]):
    if not is_logged_in(session_state):
        return [], [], render_status_html("请先登录。")
    try:
        rows, items = fetch_history_data(session_state)
        return rows, items, gr.skip()
    except ApiClientError as exc:
        return gr.skip(), gr.skip(), render_status_html(f"刷新历史失败：{exc}")


def select_history_task(
        history_state: list[dict[str, Any]],
        session_state: dict[str, Any],
        evt: gr.SelectData,
):
    if not is_logged_in(session_state):
        return (gr.skip(), render_status_html("请先登录。"), render_result_badge_html(""), DEFAULT_RESULT_BODY, gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip())
    if not history_state:
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

    raw_index = evt.index
    if isinstance(raw_index, (tuple, list)):
        if not raw_index:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        raw_index = raw_index[0]

    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return (gr.skip(), render_status_html("无法识别所选历史记录。"), render_result_badge_html("加载失败"), "请重新选择一条历史记录。", gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip())

    if index < 0 or index >= len(history_state):
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

    task_id = str(history_state[index].get("task_id", "")).strip()
    if not task_id:
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

    try:
        detail = get_api_client().get_grading_task_detail(session_state["access_token"], task_id)
        task_status_text, result_title, result_body = render_task_detail(detail)
        status = str(detail.get("status", "")).upper()
        if status in {"PENDING", "RUNNING"}:
            running_task_id = task_id
            submit_mode = "submit"
            submit_button = gr.update(value="批改中...", interactive=False)
        else:
            running_task_id = ""
            submit_mode = "next_question"
            submit_button = gr.update(value="批改下一题", interactive=True)

        return (
            task_id,
            task_status_text,
            result_title,
            result_body,
            detail.get("question_text", ""),
            detail.get("student_answer_text", ""),
            running_task_id,
            submit_mode,
            submit_button,
        )
    except ApiClientError as exc:
        return (gr.skip(), render_status_html(f"加载历史详情失败：{exc}"), render_result_badge_html("加载失败"), "无法读取该条历史记录。", gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip())


def submit_or_reset(
        session_state: dict[str, Any],
        submit_mode: str,
        question_text: str,
        student_answer_text: str,
):
    if not is_logged_in(session_state):
        return (gr.skip(), gr.skip(), render_status_html("请先登录后再提交。"), render_result_badge_html(""), DEFAULT_RESULT_BODY, "submit", "", "",
                gr.update(value="提交批改", interactive=True), gr.skip(), gr.skip())

    if submit_mode == "next_question":
        return ("", "", render_status_html(DEFAULT_TASK_STATUS), render_result_badge_html(""), DEFAULT_RESULT_BODY, "submit", "", "",
                gr.update(value="提交批改", interactive=True), gr.skip(), gr.skip())

    normalized_question = question_text.strip()
    normalized_answer = student_answer_text.strip()
    if not normalized_question or not normalized_answer:
        return (gr.skip(), gr.skip(), render_status_html("题目描述和学生答案都不能为空。"), render_result_badge_html(""), DEFAULT_RESULT_BODY,
                "submit", "", "", gr.update(value="提交批改", interactive=True), gr.skip(), gr.skip())

    try:
        client = get_api_client()
        response = client.create_grading_task(
            session_state["access_token"],
            normalized_question,
            normalized_answer,
        )
        rows, items = fetch_history_data(session_state)
        task_id = response["task_id"]
        return (
            normalized_question,
            normalized_answer,
            render_status_html(f"任务已提交\n任务 ID：{task_id}"),
            render_result_badge_html("批改中"),
            "系统正在执行多阶段批改流程。若模型长时间无响应，任务会自动结束并记录为失败。",
            "submit",
            task_id,
            task_id,
            gr.update(value="批改中...", interactive=False),
            rows,
            items,
        )
    except ApiClientError as exc:
        return (gr.skip(), gr.skip(), render_status_html(f"提交失败：{exc}"), render_result_badge_html("提交失败"), "请稍后重试。", "submit", "", "",
                gr.update(value="提交批改", interactive=True), gr.skip(), gr.skip())


def poll_running_task(
        session_state: dict[str, Any],
        running_task_id: str,
):
    if not is_logged_in(session_state) or not str(running_task_id).strip():
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

    try:
        client = get_api_client()
        detail = client.get_grading_task_detail(session_state["access_token"], running_task_id)
        task_status_text, result_title, result_body = render_task_detail(detail)
        status = str(detail.get("status", "")).upper()

        if status in {"PENDING", "RUNNING"}:
            return (task_status_text, result_title, result_body, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    gr.skip())

        rows, items = fetch_history_data(session_state)
        if status in {"SUCCEEDED", "FAILED"}:
            return (task_status_text, result_title, result_body, "next_question", "", detail.get("task_id", ""),
                    gr.update(value="批改下一题", interactive=True), rows, items)

        return (task_status_text, result_title, result_body, "submit", "", detail.get("task_id", ""),
                gr.update(value="提交批改", interactive=True), rows, items)
    except ApiClientError as exc:
        return (render_status_html(f"轮询失败：{exc}"), render_result_badge_html("网络错误"), "无法获取最新结果。", "submit", "", running_task_id,
                gr.update(value="重新提交", interactive=True), gr.skip(), gr.skip())


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GradeSystem", fill_width=True) as demo:
        session_state = gr.State(empty_session_state())
        history_state = gr.State([])
        selected_task_id_state = gr.State("")
        running_task_id_state = gr.State("")
        submit_mode_state = gr.State("submit")

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(HERO_HTML)

            user_info_markdown = gr.HTML(build_user_summary(None), elem_classes=["banner-text", "user-banner"])
            auth_feedback_markdown = gr.HTML("", elem_classes=["banner-text", "error-banner"], visible=False)

            with gr.Group(visible=True, elem_classes=["auth-stage"]) as auth_group:
                with gr.Column(scale=1, min_width=360, elem_classes=["auth-shell"]):
                    with gr.Group(elem_classes=["card-panel", "auth-card"]):
                        gr.HTML(
                            """
                            <div class="auth-title">
                              <h2>账户登录</h2>
                              <p>登录后可继续提交题目并查看个人批改记录。</p>
                            </div>
                            """
                        )
                        with gr.Tabs():
                            with gr.Tab("登录"):
                                login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                                login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                                login_button = gr.Button("登录", elem_classes=["btn-primary"])
                            with gr.Tab("注册"):
                                register_username = gr.Textbox(label="用户名", placeholder="设置用户名")
                                register_display_name = gr.Textbox(label="显示名称", placeholder="例如：王老师")
                                register_password = gr.Textbox(label="密码", type="password", placeholder="至少 8 位")
                                register_button = gr.Button("注册并登录", elem_classes=["btn-primary"])

            with gr.Group(visible=False, elem_classes=["workspace-stage"]) as workspace_group:
                with gr.Row(elem_classes=["workspace-head"]):
                    with gr.Column(scale=4):
                        gr.HTML(WORKSPACE_INTRO_HTML, elem_classes=["workspace-intro"])
                    with gr.Column(scale=2, min_width=240):
                        with gr.Row(elem_classes=["workspace-actions"]):
                            refresh_history_button = gr.Button("刷新历史", elem_classes=["btn-secondary"])
                            logout_button = gr.Button("退出登录", elem_classes=["btn-secondary"])

                with gr.Row(elem_classes=["workspace-grid"]):
                    with gr.Column(scale=3, min_width=320):
                        with gr.Group(elem_classes=["card-panel", "surface-card"]):
                            gr.HTML(render_section_title_html("历史记录"), elem_classes=["section-title"])
                            history_table = gr.Dataframe(
                                headers=["序号", "摘要", "状态", "结果", "时间"],
                                value=[],
                                interactive=False,
                                wrap=False,
                                label="",
                                max_height=480,
                                elem_classes=["history-table"],
                            )

                    with gr.Column(scale=4, min_width=360):
                        with gr.Group(elem_classes=["card-panel", "surface-card"]):
                            gr.HTML(render_section_title_html("题目与答案"), elem_classes=["section-title"])
                            question_textbox = gr.Textbox(
                                label="题目描述",
                                lines=5,
                                placeholder="请输入完整题干",
                            )
                            student_answer_textbox = gr.Textbox(
                                label="学生答案",
                                lines=5,
                                placeholder="请输入学生作答过程",
                            )
                            submit_button = gr.Button("提交批改", elem_classes=["btn-primary"])

                    with gr.Column(scale=4, min_width=360):
                        with gr.Group(elem_classes=["card-panel", "surface-card"]):
                            gr.HTML(render_section_title_html("批改结果"), elem_classes=["section-title"])
                            task_status_markdown = gr.HTML(
                                render_status_html(DEFAULT_TASK_STATUS),
                                elem_classes=["info-box", "status-box", "status-slot"],
                            )
                            result_title_markdown = gr.HTML(
                                render_result_badge_html(""),
                                elem_classes=["result-title-slot"],
                            )
                            result_body_markdown = gr.Markdown(
                                DEFAULT_RESULT_BODY,
                                elem_classes=["info-box", "result-box"],
                            )

                polling_timer = gr.Timer(2.0, active=True, render=False)

        # ====== 绑定事件 ======
        login_button.click(
            fn=login_user,
            inputs=[login_username, login_password],
            outputs=[session_state, auth_group, workspace_group, user_info_markdown, auth_feedback_markdown,
                     history_table, history_state, task_status_markdown, result_title_markdown, result_body_markdown,
                     submit_mode_state, running_task_id_state, selected_task_id_state, submit_button, login_username,
                     login_password],
        )

        register_button.click(
            fn=register_user,
            inputs=[register_username, register_display_name, register_password],
            outputs=[session_state, auth_group, workspace_group, user_info_markdown, auth_feedback_markdown,
                     history_table, history_state, task_status_markdown, result_title_markdown, result_body_markdown,
                     submit_mode_state, running_task_id_state, selected_task_id_state, submit_button, register_username,
                     register_display_name, register_password],
        )

        logout_button.click(
            fn=logout_user,
            inputs=[session_state],
            outputs=[session_state, auth_group, workspace_group, user_info_markdown, auth_feedback_markdown,
                     history_table, history_state, task_status_markdown, result_title_markdown, result_body_markdown,
                     submit_mode_state, running_task_id_state, selected_task_id_state, submit_button, question_textbox,
                     student_answer_textbox],
        )

        refresh_history_button.click(
            fn=refresh_history,
            inputs=[session_state],
            outputs=[history_table, history_state, task_status_markdown],
        )

        history_table.select(
            fn=select_history_task,
            inputs=[history_state, session_state],
            outputs=[selected_task_id_state, task_status_markdown, result_title_markdown, result_body_markdown,
                     question_textbox, student_answer_textbox, running_task_id_state, submit_mode_state, submit_button],
        )

        submit_button.click(
            fn=submit_or_reset,
            inputs=[session_state, submit_mode_state, question_textbox, student_answer_textbox],
            outputs=[question_textbox, student_answer_textbox, task_status_markdown, result_title_markdown,
                     result_body_markdown, submit_mode_state, running_task_id_state, selected_task_id_state,
                     submit_button, history_table, history_state],
        )

        polling_timer.tick(
            fn=poll_running_task,
            inputs=[session_state, running_task_id_state],
            outputs=[task_status_markdown, result_title_markdown, result_body_markdown, submit_mode_state,
                     running_task_id_state, selected_task_id_state, submit_button, history_table, history_state],
        )

    return demo


app = build_app()
