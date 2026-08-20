#!/usr/bin/env python3
"""流式清洗: 中国专利数据库YYYY年.csv -> SQLite(patents/companies/links 三表)

用法:
  python clean_patents.py --input 中国专利数据库2024年.csv --db patents.db
  python clean_patents.py --input 中国专利数据库2024年.csv --db patents.db --test 50000   # 验证前N行

策略:
  - 逐行流式读取,内存占用恒定,不把整文件读进内存
  - 只保留 发明申请 + 发明授权
  - 申请年份 >= --year-floor(默认 2016,信用代码时代)
  - 丢弃垃圾行(如宣传行)与空申请号
  - 统一社会信用代码 与 申请人 按位置配对(已验证 89.8% 对齐)
  - q_text = 标题 + 摘要 + 主权项
  - patents 以申请号为主键去重,companies 以信用代码为主键,links 为 申请号×信用代码
"""
import argparse
import csv
import io
import sqlite3
import sys

VALID_TYPES = {"发明申请", "发明授权"}

# 列名 -> 语义索引(运行时按 header 重新定位)
NEEDED = ["专利名称", "专利类型", "申请人", "申请号", "申请年份", "公开公告年份",
          "IPC分类号", "IPC主分类号", "摘要文本", "主权项内容", "当前权利人", "统一社会信用代码"]


def parse_multi(s: str):
    """按 ';' 拆分多值字段,去空白去空段。"""
    return [x.strip() for x in s.split(";") if x.strip()]


def detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(3)
    return "utf-8-sig" if head == b"\xef\xbb\xbf" else "gb18030"


def build_db(conn):
    conn.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE IF NOT EXISTS patents(
            申请号 TEXT PRIMARY KEY, 申请年份 INTEGER, 专利类型 TEXT, q_text TEXT,
            ipc_main TEXT, ipc_all TEXT, assignee_names TEXT, holder_names TEXT, 公开公告年份 INTEGER);
        CREATE TABLE IF NOT EXISTS companies(信用代码 TEXT PRIMARY KEY, 名称 TEXT);
        CREATE TABLE IF NOT EXISTS links(申请号 TEXT, 信用代码 TEXT, PRIMARY KEY(申请号, 信用代码));
        CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(名称);
        CREATE INDEX IF NOT EXISTS idx_links_cc ON links(信用代码);
        """
    )
    conn.commit()


def flush(conn, bp, bc, bl):
    conn.executemany("INSERT OR IGNORE INTO patents VALUES (?,?,?,?,?,?,?,?,?)", bp)
    conn.executemany("INSERT OR IGNORE INTO companies VALUES (?,?)", bc)
    conn.executemany("INSERT OR IGNORE INTO links VALUES (?,?)", bl)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--year-floor", type=int, default=2016)
    ap.add_argument("--test", type=int, default=0, help="只处理前 N 行(验证用)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    build_db(conn)

    fin = io.open(args.input, encoding=detect_encoding(args.input), newline="")
    r = csv.reader(fin)
    header = next(r)
    idx = {}
    for name in NEEDED:
        try:
            idx[name] = header.index(name)
        except ValueError:
            print(f"警告: 文件缺少列 {name}")
    if "申请号" not in idx:
        print("错误: 缺申请号列,终止")
        sys.exit(1)

    n_read = 0
    n_kept = 0
    n_drop = 0
    n_cc = 0
    bp, bc, bl = [], [], []
    for row in r:
        n_read += 1
        if args.test and n_read > args.test:
            break
        row = row + [""] * (len(header) - len(row))  # 短行补齐

        typ = row[idx["专利类型"]].strip()
        if typ not in VALID_TYPES:
            n_drop += 1
            continue

        pid = row[idx["申请号"]].strip()
        if not pid:
            n_drop += 1
            continue

        try:
            year = int(row[idx["申请年份"]].strip())
        except ValueError:
            n_drop += 1
            continue
        if year < args.year_floor:
            n_drop += 1
            continue

        # 申请人 与 信用代码 按位置配对;当前权利人补充无代码企业名
        app_names = parse_multi(row[idx["申请人"]])
        credits = parse_multi(row[idx["统一社会信用代码"]])
        holder_names = parse_multi(row[idx["当前权利人"]])
        pairs = []
        for i, cc in enumerate(credits):
            nm = app_names[i] if i < len(app_names) else ""
            pairs.append((cc, nm))
        for nm in holder_names:
            if nm and nm not in [p[1] for p in pairs]:
                pairs.append(("", nm))
        if credits:
            n_cc += 1

        q_text = " ".join(filter(None, [
            row[idx["专利名称"]].strip(),
            row[idx["摘要文本"]].strip(),
            row[idx["主权项内容"]].strip(),
        ]))

        bp.append((pid, year, typ, q_text,
                   row[idx["IPC主分类号"]].strip(), row[idx["IPC分类号"]].strip(),
                   ";;".join(app_names), ";;".join(holder_names),
                   row[idx["公开公告年份"]].strip() or None))
        for cc, nm in pairs:
            if cc:
                bc.append((cc, nm))
                bl.append((pid, cc))
        n_kept += 1

        if len(bp) >= 50000:
            flush(conn, bp, bc, bl)
            bp, bc, bl = [], [], []
    flush(conn, bp, bc, bl)
    fin.close()
    conn.commit()
    conn.close()
    print(f"读取={n_read} 保留发明={n_kept} 丢弃={n_drop} 含信用代码行={n_cc}")


if __name__ == "__main__":
    main()
