"""从 企业数据/*.xls 抽取 所属行业/企业类型/注册地址/经营状态，与 g_company_pool.csv 合并成 g_company_info.csv。

读取 .xls 用 olefile + xlrd 的 file_contents= 变通（xlrd 1.2.0 对这批 CFB 有 compdoc bug）。
xls 列：0企业名称 1经营状态 15统一社会信用代码 20企业类型 21所属行业 23注册地址 26经营范围 27匹配状态。
只保留 匹配状态=成功 的行，按信用代码匹配池，池里有但 xls 缺的留空并告警。
"""
import csv
import glob
import os

import olefile
import xlrd

XLS_DIR = os.path.join(os.path.dirname(__file__), '企业数据')
POOL = os.path.join(os.path.dirname(__file__), 'g_company_pool.csv')
OUT = os.path.join(os.path.dirname(__file__), 'g_company_info.csv')

HEADER = ['信用代码', '名称', '类型', '专利数', '经营范围', '所属行业', '企业类型', '注册地址', '经营状态']


def read_xls(path):
    ole = olefile.OleFileIO(path)
    return xlrd.open_workbook(file_contents=ole.openstream('Workbook').read()).sheet_by_index(0)


def main():
    info = {}  # code -> dict of fields
    for path in sorted(glob.glob(os.path.join(XLS_DIR, '*.xls'))):
        sh = read_xls(path)
        for r in range(3, sh.nrows):
            if str(sh.cell_value(r, 27)).strip() != '成功':
                continue
            code = str(sh.cell_value(r, 15)).strip()
            info[code] = {
                '经营范围': str(sh.cell_value(r, 26)).strip(),
                '所属行业': str(sh.cell_value(r, 21)).strip(),
                '企业类型': str(sh.cell_value(r, 20)).strip(),
                '注册地址': str(sh.cell_value(r, 23)).strip(),
                '经营状态': str(sh.cell_value(r, 1)).strip(),
            }

    with open(POOL, encoding='utf-8-sig', newline='') as f:
        pool = list(csv.reader(f))[1:]

    rows = [HEADER]
    missing = []
    for code, name, patent_cnt, typ in pool:
        if code not in info:
            missing.append((code, name))
            info[code] = {'经营范围': '', '所属行业': '', '企业类型': '', '注册地址': '', '经营状态': ''}
        d = info[code]
        rows.append([code, name, typ, patent_cnt, d['经营范围'], d['所属行业'], d['企业类型'], d['注册地址'], d['经营状态']])

    with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)

    n_empty = sum(1 for r in rows[1:] if not r[4])
    print(f'g_company_info.csv: {len(rows) - 1} 家')
    print(f'经营范围空: {n_empty}')
    print(f'池里有但 xls 找不到(缺失): {len(missing)}')
    for code, name in missing[:10]:
        print('  缺失:', code, name)


if __name__ == '__main__':
    main()
