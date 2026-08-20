"""把爱企查批量查询结果(企业数据/*.xls)的经营范围回填进 g_company_scope_fill.csv，并清理问题实体。

读取 .xls 用 olefile + xlrd 的 file_contents= 变通（xlrd 1.2.0 对这批 CFB 有 compdoc bug）。

清理规则（用户定案「全部严格清理」）：
  1. 匹配状态 == 失败                    -> 删
  2. 池里有但任何 xls 都没有              -> 删（未查询）
  3. 成功但 经营状态 == 删除              -> 删
  4. 成功但 经营范围 空/未公示/-          -> 删
  5. 池「名称」与爱企查「名称」相似度<=0.5 -> 删（疑似同名挂错码/码指向别家）
其余保留并回填经营范围（按信用代码匹配）。不写备份。
"""
import csv
import difflib
import glob
import os

import olefile
import xlrd

XLS_DIR = os.path.join(os.path.dirname(__file__), '企业数据')
POOL = os.path.join(os.path.dirname(__file__), 'g_company_pool.csv')
SCOPE = os.path.join(os.path.dirname(__file__), 'g_company_scope_fill.csv')

EMPTY_SCOPE = {'', '-', '未公示', 'None'}


def read_xls(path):
    ole = olefile.OleFileIO(path)
    return xlrd.open_workbook(file_contents=ole.openstream('Workbook').read()).sheet_by_index(0)


def norm(s):
    return s.replace('（', '(').replace('）', ')').replace(' ', '').replace('　', '')


def main():
    # 1. 从 xls 提取 (信用代码 -> 记录)，并标记删除原因
    valid = {}        # code -> dict(scope=..., xls_name=...)
    drop_reason = {}  # code -> reason str
    for path in sorted(glob.glob(os.path.join(XLS_DIR, '*.xls'))):
        sh = read_xls(path)
        for r in range(3, sh.nrows):
            match = str(sh.cell_value(r, 27)).strip()
            code = str(sh.cell_value(r, 15)).strip() if match == '成功' else str(sh.cell_value(r, 0)).strip()
            name = str(sh.cell_value(r, 0)).strip()
            scope = str(sh.cell_value(r, 26)).strip()
            status = str(sh.cell_value(r, 1)).strip()
            if match == '失败':
                drop_reason[code] = '查询失败'
            elif status == '删除':
                drop_reason[code] = '经营状态=删除'
            elif scope in EMPTY_SCOPE:
                drop_reason[code] = '经营范围空'
            else:
                valid[code] = {'scope': scope, 'xls_name': name}

    # 2. 读池与 scope-fill
    with open(POOL, encoding='utf-8-sig', newline='') as f:
        pool = list(csv.reader(f))
    pool_header, pool_rows = pool[0], pool[1:]

    # 池里未出现在任何 xls 的 -> 未查询
    for code, _, _, _ in pool_rows:
        if code not in valid and code not in drop_reason:
            drop_reason[code] = '未查询'

    # 3. 生成清理后的池 + 回填经营范围
    new_pool = [pool_header]
    new_scope = [['信用代码', '名称', '类型', '经营范围']]
    dropped = []
    for code, name, patent_cnt, typ in pool_rows:
        if code not in valid:
            dropped.append((code, name, typ, drop_reason.get(code, '?')))
            continue
        xls_name = valid[code]['xls_name']
        # 名称不一致且相似度低 -> 疑似挂错码
        if norm(name) != norm(xls_name) and difflib.SequenceMatcher(None, norm(name), norm(xls_name)).ratio() <= 0.5:
            dropped.append((code, name, typ, '名称与爱企查不一致'))
            continue
        new_pool.append([code, name, patent_cnt, typ])
        new_scope.append([code, name, typ, valid[code]['scope']])

    with open(POOL, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(new_pool)
    with open(SCOPE, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(new_scope)

    print(f'保留 {len(new_pool) - 1} 家，本轮删除 {len(dropped)} 家')
    for code, name, typ, reason in dropped:
        print(f'  删: {code} {name} [{typ}] -> {reason}')


if __name__ == '__main__':
    main()
