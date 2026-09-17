"""从已经保存的三个原始年报提取全部页面，禁止覆盖现有提取记录。"""
from pathlib import Path
import sys
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.eps_disclosed_holdings_diagnostic_v1 import OUT, identity, read, save


def main() -> None:
    collection=read(OUT/"source_collection_result.json")
    targets=[OUT/f"annual_{row['year']}_pages.json" for row in collection["documents"]]
    if any(p.exists() for p in targets):
        raise FileExistsError("页面提取对象已存在；验证请运行离线核对脚本，不覆盖旧记录")
    for document,target in zip(collection["documents"],targets):
        original=ROOT/document["pdf"]["path"]
        if identity(original)!=document["pdf"]:
            raise ValueError("原件哈希变化")
        with pdfplumber.open(original) as pdf:
            pages=[{"page":i+1,"text":page.extract_text() or ""} for i,page in enumerate(pdf.pages)]
            save(target,{"year":document["year"],"source":document["pdf"],"metadata":pdf.metadata,"pages":pages})
        print(f"{document['year']}年完整页面已保存。",flush=True)


if __name__=="__main__":
    main()
