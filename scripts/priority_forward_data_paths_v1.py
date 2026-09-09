"""识别本项目已批准的数据 Junction，并保留可移植的逻辑相对路径。"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPROVED_DATA_ROOT = Path(r"E:\ResearchData\New project 8\data")


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path = ROOT
    approved_data_root: Path = APPROVED_DATA_ROOT

    def validate_data_junction(self) -> dict[str, str]:
        logical = self.root / "data"
        details = logical.lstat()
        if getattr(details, "st_reparse_tag", None) != stat.IO_REPARSE_TAG_MOUNT_POINT:
            raise ValueError("项目 data 必须是已批准的 Windows Junction")
        approved = self.approved_data_root
        if not approved.is_absolute() or approved.resolve(strict=True) != approved:
            raise ValueError("批准的数据目录必须对应固定的实际绝对目录")
        if logical.resolve(strict=True) != approved:
            raise ValueError("data Junction 的实际目标与已批准的 E 盘目录不一致")
        return {"logical_data_root": str(logical), "resolved_data_root": str(approved)}

    def checked(self, value: str | Path) -> Path:
        """校验实际落点，返回项目内逻辑路径供旧回执保持相对路径格式。"""
        raw = Path(value)
        if ".." in raw.parts:
            raise ValueError("受控路径不允许包含上级目录跳转")
        root = self.root.absolute()
        logical = raw if raw.is_absolute() else root / raw
        try:
            relative = logical.relative_to(root)
        except ValueError as exc:
            raise ValueError("仅接受本项目逻辑目录内的路径") from exc
        resolved = logical.resolve()
        if relative.parts and relative.parts[0].casefold() == "data":
            self.validate_data_junction()
            expected = self.approved_data_root.joinpath(*relative.parts[1:])
            if resolved != expected:
                raise ValueError("data 下的路径出现额外重定向，实际落点与声明不一致")
        else:
            try:
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError("非 data 路径不得离开项目实际目录") from exc
        return logical

    def relative(self, value: str | Path) -> str:
        return self.checked(value).relative_to(self.root.absolute()).as_posix()


PATHS = WorkspacePaths()
