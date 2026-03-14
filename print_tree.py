"""
print_tree.py
放到 BallonsTranslator 根目錄執行：
    python print_tree.py
輸出專案檔案樹狀圖到 project_tree.txt
"""

from pathlib import Path

IGNORE_DIRS = {
    '__pycache__', '.git', '.github', '.venv', 'venv',
    'node_modules', '.mypy_cache', '.pytest_cache',
    'dist', 'build', '.eggs',
    # 打包環境，跟程式碼無關
    'ballontrans_pylibs_win', 'ballontrans_pylibs_linux',
    'ballontrans_pylibs_mac', 'PortableGit',
}

IGNORE_EXTS = {
    '.pyc', '.pyo', '.pyd',
    '.pt', '.pth', '.onnx', '.bin', '.safetensors',
    '.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.db', '.sqlite', '.log',
}

IGNORE_FILES = {'.DS_Store', 'Thumbs.db', 'desktop.ini'}


def should_skip(entry: Path) -> bool:
    if entry.is_dir():
        return entry.name in IGNORE_DIRS or entry.name.endswith('.egg-info')
    return entry.name in IGNORE_FILES or entry.suffix.lower() in IGNORE_EXTS


def build_tree(root: Path, prefix: str = '', output: list = None) -> list:
    if output is None:
        output = []
    try:
        entries = sorted(
            [e for e in root.iterdir() if not should_skip(e)],
            key=lambda e: (e.is_file(), e.name.lower())
        )
    except PermissionError:
        output.append(f"{prefix}[無法存取]")
        return output

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = '└── ' if is_last else '├── '
        output.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            build_tree(entry, prefix + ('    ' if is_last else '│   '), output)

    return output


def main():
    root = Path(__file__).parent
    print(f"掃描目錄：{root}")

    lines = [str(root.resolve()) + '/']
    build_tree(root, '', lines)

    output_path = root / 'project_tree.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    size_kb = output_path.stat().st_size / 1024
    print(f"完成！共 {len(lines)} 行，{size_kb:.1f} KB → {output_path}")
    if size_kb > 100:
        print("⚠ 仍偏大，可把 data/ 也加入 IGNORE_DIRS")


if __name__ == '__main__':
    main()
