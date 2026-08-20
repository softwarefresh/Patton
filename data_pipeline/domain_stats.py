#!/usr/bin/env python3
"""按 IPC 领域统计母库,供选领域用。

输出(按 IPC 段 A-H):
  - 该段专利数(母库)
  - 该段内企业数(持有该段专利的去重企业)
  - 该段内"候选池企业"数(在该段持有>=5项专利)
  - 该段内"查询候选"数(申请年份 2022-2024 且正例在候选池内)
  - 细分:top IPC 小类(前4位)专利数
"""
import sqlite3
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data_pipeline/patents.db")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect(args.db)

print("--- IPC 主分类格式抽查 ---", flush=True)
for r in conn.execute("SELECT ipc_main FROM patents WHERE ipc_main!='' LIMIT 5"):
    print("  ", repr(r[0]), flush=True)

print("\n--- 各 IPC 段专利数 ---", flush=True)
sec_pat = dict(conn.execute(
    "SELECT substr(ipc_main,1,1) sec, COUNT(*) FROM patents "
    "WHERE ipc_main!='' GROUP BY sec").fetchall())
for s, c in sorted(sec_pat.items()):
    print(f"  {s}: {c:,} 项专利", flush=True)

print("\n--- 各 IPC 段查询候选(2022-2024申请, 正例在候选池) ---", flush=True)
sec_q = {}
for s, c in conn.execute(
    "SELECT substr(p.ipc_main,1,1), COUNT(DISTINCT p.申请号) "
    "FROM patents p JOIN links l ON l.申请号=p.申请号 "
    "WHERE p.申请年份 BETWEEN 2022 AND 2024 "
    "GROUP BY 1").fetchall():
    sec_q[s] = c
for s in sorted(sec_q):
    print(f"  {s}: {sec_q[s]:,} 个查询候选", flush=True)

print("\n--- 各 IPC 段内候选池企业数(该段持有>=5专利) ---", flush=True)
sec_comp = {}
for s, c in conn.execute(
    "SELECT sec, COUNT(*) FROM ("
    "  SELECT substr(p.ipc_main,1,1) sec, l.信用代码 "
    "  FROM links l JOIN patents p ON p.申请号=l.申请号 "
    "  GROUP BY sec, l.信用代码 HAVING COUNT(*)>=5) "
    "GROUP BY sec").fetchall():
    sec_comp[s] = c
for s in sorted(sec_comp):
    print(f"  {s}: {sec_comp[s]:,} 家", flush=True)

print("\n--- top20 IPC 小类(前4位) ---", flush=True)
for s, c in conn.execute(
    "SELECT substr(ipc_main,1,4) sub, COUNT(*) FROM patents "
    "WHERE ipc_main!='' GROUP BY 1 ORDER BY 2 DESC LIMIT 20").fetchall():
    print(f"  {s}: {c:,}", flush=True)

print("\n--- 汇总 ---", flush=True)
for s in sorted(set(sec_pat) | set(sec_q) | set(sec_comp)):
    print(f"  段{s}: 专利 {sec_pat.get(s,0):,} | 查询候选 {sec_q.get(s,0):,} | 候选池企业 {sec_comp.get(s,0):,}",
          flush=True)
conn.close()
