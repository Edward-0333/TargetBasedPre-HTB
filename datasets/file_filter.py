from pathlib import Path


def main() -> None:
    processed_root = Path("./train/processed")
    data_root = Path("./train/data")

    if not processed_root.exists() or not data_root.exists():
        return

    def build_key(root: Path, path: Path):
        relative = path.relative_to(root)
        parent = relative.parent.as_posix()
        return parent, relative.stem

    processed_keys = {
        build_key(processed_root, path)
        for path in processed_root.rglob("*")
        if path.is_file()
    }

    for data_path in data_root.rglob("*"):
        if not data_path.is_file():
            continue
        if build_key(data_root, data_path) not in processed_keys:
            data_path.unlink()


if __name__ == "__main__":
    main()
