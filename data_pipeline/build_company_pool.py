#!/usr/bin/env python3
"""从 patents.db 构建企业池并标注类型,导出「经营范围补全清单」。

输出: data_pipeline/company_pool.csv (信用代码, 名称, 专利数, 类型)

类型启发式(按名称关键词):
  高校       : 大学 / 学院 / 学校 / 党校 / 广播电视大学
  科研院所   : 研究院 / 研究所 / 科学院 / 研究中心 / 测试中心 / 检测中心 / 计量院 / 勘察设计院
  事业单位/医院: 医院 / 疾控 / 血站 / 中心(如某管理中心)
  军队/政府机构: 中国人民解放军 / 部队 / 人民政府 / 人民法院 / 人民检察院 / 公安局 / 部队院校
  企业       : 其余(含"有限公司/集团/厂/股份有限公司"等)

用法:
  python data_pipeline/build_company_pool.py --db data_pipeline/patents.db \
      --min-patents 5 --out data_pipeline/company_pool.csv
"""
import argparse
import csv
import sqlite3
import sys

# 顺序即优先级:军队/政府 最优先(如"XX海关""公安局"不能落入其他类)
KEYWORDS = [
    ("军队/政府", ["中国人民解放军", "军区", "部队", "军事", "公安局", "派出所", "人民检察院",
                  "人民法院", "人民政府", "海关", "出入境", "税务局", "税务", "质量监督",
                  "气象局", "地震局", "地质调查局", "生态环境", "监督管理局", "管理局",
                  "委员会办公室", "口岸", "总站", "武警", "消防"]),
    ("事业单位/医院", ["医院", "疾控中心", "血站", "血液中心", "管理中心", "服务中心",
                     "社区卫生", "卫生院", "检验检疫", "血检"]),
    ("科研院所", ["研究院", "研究所", "科学院", "研究中心", "研究总院",
                "测试中心", "检测中心", "计量院", "勘察设计研究院"]),
    ("高校", ["大学", "学院", "学校", "党校", "广播电视大学", "职业技术学院", "警官学院"]),
]


def classify(name: str) -> str:
    for typ, kws in KEYWORDS:
        for kw in kws:
            if kw in name:
                return typ
    return "企业"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-patents", type=int, default=5)
    ap.add_argument("--out", default="data_pipeline/company_pool.csv")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    # 给 companies 表加 type 列(幂等)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(companies)")]
    if "type" not in cols:
        conn.execute("ALTER TABLE companies ADD COLUMN type TEXT DEFAULT '企业'")

    # 公司专利数(用 links 统计)
    n = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    print(f"公司总数: {n}", flush=True)
    rows = conn.execute(
        """
        SELECT c.信用代码, c.名称, COUNT(l.申请号) AS cnt
        FROM companies c JOIN links l ON l.信用代码 = c.信用代码
        GROUP BY c.信用代码 HAVING cnt >= ?
        """,
        (args.min_patents,),
    ).fetchall()
    print(f"专利数>={args.min_patents} 的公司: {len(rows)}", flush=True)

    # 分类并写库
    from collections import Counter
    type_cnt = Counter()
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["信用代码", "名称", "专利数", "类型"])
        for cc, name, cnt in rows:
            typ = classify(name)
            type_cnt[typ] += 1
            conn.execute("UPDATE companies SET type=? WHERE 信用代码=?", (typ, cc))
            w.writerow([cc, name, cnt, typ])
    conn.commit()
    conn.close()
    print("类型分布:", dict(type_cnt), flush=True)
    print(f"已导出: {args.out}", flush=True)


if __name__ == "__main__":
    main()
