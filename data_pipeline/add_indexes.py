"""给 g_domain.db 的 links_g / patents_g 加查询索引（一次性）。"""
import sqlite3
import time

con = sqlite3.connect('data_pipeline/g_domain.db')
cur = con.cursor()
for idx, table, col in [
    ('idx_lg_code', 'links_g', '信用代码'),
    ('idx_lg_appn', 'links_g', '申请号'),
    ('idx_pg_year', 'patents_g', '申请年份'),
]:
    t = time.time()
    cur.execute(f'CREATE INDEX IF NOT EXISTS {idx} ON {table}({col})')
    con.commit()
    print(f'{idx} 建好, 耗时 {time.time()-t:.0f}s')
print('索引全部完成')
con.close()
