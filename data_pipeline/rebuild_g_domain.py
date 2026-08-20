"""从母库 patents.db 重建 G 段工作库 g_domain_new.db（损坏恢复用）。

与 build_g_domain.py 的复制逻辑一致，但:
  - 使用默认 DELETE journal 模式（原库 journal_mode=OFF 导致杀进程即损坏）
  - 只复制 patents_g / links_g / companies_g 三张表，不重建候选池（池 CSV 已定稿）
  - 建到 g_domain_new.db，校验行数后再手动替换
"""
import sqlite3
import sys
import time

MASTER = 'data_pipeline/patents.db'
OUT = 'data_pipeline/g_domain_new.db'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    m = sqlite3.connect(MASTER)
    g = sqlite3.connect(OUT)
    g.executescript('''
        CREATE TABLE IF NOT EXISTS patents_g(
            申请号 TEXT PRIMARY KEY, 申请年份 INTEGER, q_text TEXT,
            ipc_main TEXT, ipc_all TEXT, assignee_names TEXT, holder_names TEXT);
        CREATE TABLE IF NOT EXISTS links_g(申请号 TEXT, 信用代码 TEXT, PRIMARY KEY(申请号, 信用代码));
        CREATE TABLE IF NOT EXISTS companies_g(信用代码 TEXT PRIMARY KEY, 名称 TEXT, type TEXT);
    ''')
    m.execute('ATTACH DATABASE ? AS g', (OUT,))

    print('复制 G 段专利 ...', flush=True)
    m.execute('INSERT OR IGNORE INTO g.patents_g '
              'SELECT 申请号, 申请年份, q_text, ipc_main, ipc_all, assignee_names, holder_names '
              "FROM patents WHERE ipc_main LIKE 'G%'")
    m.commit()
    print(f'  patents_g 完成, 耗时 {time.time()-t0:.0f}s', flush=True)

    print('复制 G 段链接 ...', flush=True)
    m.execute('INSERT OR IGNORE INTO g.links_g '
              'SELECT l.申请号, l.信用代码 FROM links l '
              'JOIN g.patents_g p ON p.申请号 = l.申请号')
    m.commit()
    print(f'  links_g 完成, 耗时 {time.time()-t0:.0f}s', flush=True)

    m.execute('INSERT OR IGNORE INTO g.companies_g SELECT 信用代码, 名称, type FROM companies')
    m.commit()
    m.execute('DETACH DATABASE g')
    m.close()

    np = g.execute('SELECT COUNT(*) FROM patents_g').fetchone()[0]
    nl = g.execute('SELECT COUNT(*) FROM links_g').fetchone()[0]
    nc = g.execute('SELECT COUNT(*) FROM companies_g').fetchone()[0]
    print(f'校验: patents_g={np:,} links_g={nl:,} companies_g={nc:,}', flush=True)
    print('quick_check:', str(g.execute('PRAGMA quick_check').fetchone()[0])[:100], flush=True)
    g.close()
    print(f'全部完成, 总耗时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
