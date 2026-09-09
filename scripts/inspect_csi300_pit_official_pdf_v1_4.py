from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterable

import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import (  # noqa: E402
    csi300_pit_fundamental_underreaction_official_facts_v1 as _base,
)
from research import (  # noqa: E402
    csi300_pit_fundamental_underreaction_official_facts_v1_3 as _v1_3,
)


OCR_SCRIPT = ROOT / "scripts/windows_ocr_financial_statement_v1_3.ps1"


def _parse_pages(expression: str, page_count: int) -> tuple[int, ...]:
    pages: set[int] = set()
    for item in expression.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"页码范围倒置：{token}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(token))
    invalid = sorted(page for page in pages if page < 1 or page > page_count)
    if invalid:
        raise ValueError(f"页码超出范围：{invalid}；总页数={page_count}")
    return tuple(sorted(page - 1 for page in pages))


def _matching_pages(
    page_texts: Iterable[str],
    patterns: Iterable[str],
) -> tuple[int, ...]:
    compiled = [re.compile(pattern) for pattern in patterns]
    return tuple(
        index
        for index, text in enumerate(page_texts)
        if any(pattern.search(text) for pattern in compiled)
    )


def _render_pages(
    content: bytes,
    pages: Iterable[int],
    output_root: Path,
    scale: float,
    crop: tuple[float, float, float, float] | None = None,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(content)
    outputs: list[Path] = []
    try:
        for page_index in pages:
            page = document[page_index]
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                crop_suffix = ""
                if crop is not None:
                    left, top, right, bottom = crop
                    image = image.crop(
                        (
                            round(image.width * left),
                            round(image.height * top),
                            round(image.width * right),
                            round(image.height * bottom),
                        )
                    )
                    crop_suffix = "_crop_" + "_".join(f"{value:g}" for value in crop)
                output_path = output_root / (
                    f"page_{page_index + 1:04d}_x{scale:g}{crop_suffix}.png"
                )
                image.save(output_path)
                outputs.append(output_path)
            finally:
                page.close()
    finally:
        document.close()
    return outputs


def _run_ocr(image_paths: Iterable[Path]) -> None:
    paths = [str(path.resolve()) for path in image_paths]
    if not paths:
        return
    with tempfile.TemporaryDirectory(prefix="csi300_v1_4_inspect_ocr_") as temporary:
        list_path = Path(temporary) / "images.json"
        list_path.write_text(
            json.dumps(paths, ensure_ascii=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(OCR_SCRIPT),
                "-ImageListPath",
                str(list_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Windows OCR失败："
            f"exit={completed.returncode}；{completed.stderr[-2000:]}"
        )
    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(records) != len(paths):
        raise RuntimeError(f"Windows OCR页数不一致：{len(records)} != {len(paths)}")
    for record in records:
        print(f"\n===== OCR｜{Path(record['image_path']).name} =====")
        for line_number, line in enumerate(record.get("lines") or [], start=1):
            print(f"{line_number:04d}: {line.get('text') or ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查V1.4固定重放PDF的文本层与关键版面")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="")
    parser.add_argument("--find", action="append", default=[])
    parser.add_argument("--engine", choices=("pdfium", "pdfplumber"), default="pdfium")
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--scale", type=float, default=2.25)
    parser.add_argument(
        "--crop",
        help="按页面宽高比例裁剪，格式为left,top,right,bottom，均在0到1之间",
    )
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr-summary", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    content = pdf_path.read_bytes()
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"输入不是PDF：{pdf_path}")
    if args.engine == "pdfplumber":
        page_texts, errors = _v1_3._extract_pdfplumber_page_texts(content)
        if errors:
            print(f"PDFPLUMBER页错误：{errors}")
    else:
        page_texts, receipt = _base.extract_pdf_page_texts(content)
        print(f"PDFIUM文本回执：{receipt}")

    pages = _parse_pages(args.pages, len(page_texts)) if args.pages else ()
    if args.find:
        pages = tuple(sorted(set(pages).union(_matching_pages(page_texts, args.find))))
    if not pages:
        raise ValueError("必须通过--pages或--find指定至少一页")

    for page_index in pages:
        text = page_texts[page_index]
        print(f"\n===== 第{page_index + 1}页｜字符数={len(text)} =====")
        for line_number, line in enumerate(text.splitlines(), start=1):
            print(f"{line_number:04d}: {line}")

    if args.tables:
        rows, errors = _base.extract_pdf_table_rows(content, pages)
        print(f"\nPDF表格行数：{len(rows)}；页错误：{errors}")
        for page_index, table_index, row_index, cells in rows:
            normalized = [_base.normalize_text(cell) for cell in cells]
            if any(normalized):
                print(
                    f"第{page_index + 1}页/表{table_index}/行{row_index}: "
                    f"{normalized}"
                )

    crop = None
    if args.crop:
        values = tuple(float(value) for value in args.crop.split(","))
        if len(values) != 4 or not (
            0 <= values[0] < values[2] <= 1
            and 0 <= values[1] < values[3] <= 1
        ):
            raise ValueError("--crop必须是合法的left,top,right,bottom比例")
        crop = values
    if args.render_dir:
        outputs = _render_pages(
            content,
            pages,
            args.render_dir.resolve(),
            args.scale,
            crop,
        )
        for output in outputs:
            print(f"已渲染：{output}")
        if args.ocr:
            _run_ocr(outputs)
    elif args.ocr:
        raise ValueError("--ocr必须与--render-dir同时使用")
    if args.ocr_summary:
        outputs, receipt = _v1_3._run_windows_ocr(content, pages)
        by_scale = {
            scale: _v1_3._ocr_observations(records)
            for scale, records in outputs.items()
        }
        print(f"\n双尺度OCR回执：{receipt}")
        for metric_id in _v1_3.OCR_ROW_PATTERNS:
            agreed = _v1_3._agreed_ocr_observation(by_scale, metric_id)
            if agreed is None:
                candidates = {
                    str(scale): [
                        {
                            "page": item.page_index + 1,
                            "value": str(item.value),
                            "row": item.row_text,
                        }
                        for item in observations.get(metric_id) or []
                    ]
                    for scale, observations in by_scale.items()
                }
                print(f"{metric_id}: 未达成双尺度一致｜{candidates}")
            else:
                print(
                    f"{metric_id}: 第{agreed[1].page_index + 1}页｜"
                    f"{agreed[1].value}｜{agreed[1].row_text}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
