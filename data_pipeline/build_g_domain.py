#!/usr/bin/env python3
"""构建 G 段(物理)工作子库 + 企业池清单。

步骤:
1. 从母库复制 G 段专利(ipc_main LIKE 'G%')到 g_domain.db
2. 复制对应链接
3. 报告 G 段企业池在若干阈值下的规模(便于你选)
4. 按 --pool-threshold 建池,导出经营范围补全清单 g_company_pool.csv
5. 报告 G 段查询候选数(2022-2024 申请,正例在池内)
"""
import argparse
import csv
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data_pipeline/patents.db")
    ap.add_argument("--out-db", default="data_pipeline/g_domain.db")
    ap.add_argument("--pool-threshold", type=int, default=5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    m = sqlite3.connect(args.master)
    g = sqlite3.connect(args.out_db)
    g.executescript(
        """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE IF NOT EXISTS patents_g(
            申请号 TEXT PRIMARY KEY, 申请年份 INTEGER, q_text TEXT,
            ipc_main TEXT, ipc_all TEXT, assignee_names TEXT, holder_names TEXT);
        CREATE TABLE IF NOT EXISTS links_g(申请号 TEXT, 信用代码 TEXT, PRIMARY KEY(申请号, 信用代码));
        CREATE TABLE IF NOT EXISTS companies_g(信用代码 TEXT PRIMARY KEY, 名称 TEXT, type TEXT);
        CREATE INDEX IF NOT EXISTS idx_links_g_cc ON links_g(信用代码);
        """
    )
    m.execute("ATTACH DATABASE ? AS g", (args.out_db,))

    print("复制 G 段专利 ...", flush=True)
    m.execute(
        "INSERT OR IGNORE INTO g.patents_g "
        "SELECT 申请号, 申请年份, q_text, ipc_main, ipc_all, assignee_names, holder_names "
        "FROM patents WHERE ipc_main LIKE 'G%'"
    )
    m.commit()
    print("复制 G 段链接 ...", flush=True)
    m.execute(
        "INSERT OR IGNORE INTO g.links_g "
        "SELECT l.申请号, l.信用代码 FROM links l JOIN g.patents_g p ON p.申请号 = l.申请号"
    )
    m.commit()
    m.execute("DETACH DATABASE g")
    m.close()

    np = g.execute("SELECT COUNT(*) FROM patents_g").fetchone()[0]
    nl = g.execute("SELECT COUNT(*) FROM links_g").fetchone()[0]
    print(f"G 段专利: {np:,} | G 段链接: {nl:,}", flush=True)

    print("G 段企业池规模(按持有 G 专利数阈值):", flush=True)
    for t in (2, 3, 5, 10, 20):
        n = g.execute(
            "SELECT COUNT(*) FROM (SELECT 信用代码 FROM links_g GROUP BY 信用代码 HAVING COUNT(*)>=?)",
            (t,),
        ).fetchone()[0]
        print(f"  >= {t} 项: {n:,} 家", flush=True)

    g.execute("DELETE FROM companies_g")
    g.commit()
    g.execute("ATTACH DATABASE ? AS m", (args.master,))
    g.execute(
        """
        INSERT OR IGNORE INTO companies_g
        SELECT l.信用代码, c.名称, c.type
        FROM (SELECT 信用代码, COUNT(*) cnt FROM links_g GROUP BY 信用代码 HAVING cnt>=?) l
        JOIN m.companies c ON c.信用代码 = l.信用代码
        """,
        (args.pool_threshold,),
    )
    g.commit()
    g.execute("DETACH DATABASE m")
    nc = g.execute("SELECT COUNT(*) FROM companies_g").fetchone()[0]
    print(f"G 段企业池(阈值>={args.pool_threshold}): {nc:,} 家", flush=True)

    # 一次性预计算各公司 G 专利数(避免每行相关子查询全表扫描)
    g.execute("CREATE TEMP TABLE IF NOT EXISTS cnt AS SELECT 信用代码, COUNT(*) AS c FROM links_g GROUP BY 信用代码")
    with open("data_pipeline/g_company_pool.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["信用代码", "名称", "专利数", "类型"])
        for row in g.execute(
            "SELECT c.信用代码, c.名称, n.c, c.type "
            "FROM companies_g c LEFT JOIN cnt n ON n.信用代码 = c.信用代码 "
            "ORDER BY n.c DESC"
        ):
            w.writerow(row)
    print("已导出 data_pipeline/g_company_pool.csv", flush=True)

    nq = g.execute(
        """
        SELECT COUNT(DISTINCT p.申请号) FROM patents_g p
        JOIN links_g l ON l.申请号 = p.申请号
        JOIN companies_g c ON c.信用代码 = l.信用代码
        WHERE p.申请年份 BETWEEN 2022 AND 2024
        """
    ).fetchone()[0]
    print(f"G 段查询候选(2022-2024, 正例在池内): {nq:,}", flush=True)
    g.close()


if __name__ == "__main__":
    main()
