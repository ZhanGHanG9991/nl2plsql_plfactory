import re

def extract_plsql_from_tags(text: str) -> str:
    if not text:
        return ""
    
    cleaned = text.strip()
    TAG_VARIANT = r"(?:pl)?(?:pg)?sql"
    
    # 方法1: Markdown 代码块 (优先级最高)
    md_pattern = r"```(?:sql|plsql|pgsql|plpgsql|pl/?sql|pl/?pgsql)?\s*\n?(.*?)```"
    match_md = re.search(md_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match_md:
        content = match_md.group(1).strip()
        if content:
            return _normalize_plsql_ending(content)
    
    # 方法2: XML 标签提取
    start_tag_pattern = rf"<start-{TAG_VARIANT}>"
    match_start = re.search(start_tag_pattern, cleaned, re.IGNORECASE)
    if match_start:
        content_start = match_start.end()
        remaining = cleaned[content_start:]
        end_tag_pattern = rf"</?(?:end|start)-{TAG_VARIANT}>"
        match_end = re.search(end_tag_pattern, remaining, re.IGNORECASE)
        if match_end:
            content = remaining[:match_end.start()].strip()
        else:
            content = remaining.strip()
        if content:
            return _normalize_plsql_ending(content)
    
    # 方法3: 基于关键字截断
    keyword_pattern = r'(?:^|\n)\s*((?:CREATE(?:\s+OR\s+REPLACE)?|DECLARE|BEGIN|DO)\s+.*)'
    match_keyword = re.search(keyword_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match_keyword:
        content = match_keyword.group(1).strip()
        return _normalize_plsql_ending(content)
    
    return _normalize_plsql_ending(cleaned)


def _normalize_plsql_ending(content: str) -> str:
    if not content:
        return ""
    content = content.strip()
    TAG_VARIANT = r"(?:pl)?(?:pg)?sql"
    content = re.sub(rf'\s*</?(?:end-|start-)?{TAG_VARIANT}>\s*$', '', content, flags=re.IGNORECASE).strip()
    content = re.sub(r'\s*```\s*$', '', content).strip()
    if re.search(r'\$\$\s*LANGUAGE\s+\w+\s*$', content, re.IGNORECASE):
        content = content.rstrip() + ';'
    return content

def post_process_plsql(plsql: str) -> str:
    '''
    后处理 PL/SQL 代码，去除注释。
    兼容 PostgreSQL PL/pgSQL 和 Oracle PL/SQL。
    
    功能：
    - 去除 -- 单行注释
    - 去除 /* ... */ 块注释（支持 PostgreSQL 嵌套块注释）
    - 保留字符串内的注释标记
    - 支持 PL/pgSQL Dollar-quoted 字符串 ($$...$$, $tag$...$tag$)
    - 支持 Oracle Q-quoted 字符串 (q'[...]', q'{...}' 等)
    - 支持 PostgreSQL E-string (E'...') 转义字符串
    
    Args:
        plsql: 原始 PL/SQL 代码
    
    Returns:
        处理后的 PL/SQL 代码
    '''
    if not plsql:
        return plsql
    
    result = []
    i = 0
    n = len(plsql)
    
    while i < n:
        # 1. 检查块注释 /* ... */
        if i + 1 < n and plsql[i:i+2] == '/*':
            # 跳过块注释 (支持嵌套)
            end_idx = _find_block_comment_end(plsql, i)
            i = end_idx
            continue
        
        # 2. 检查单行注释 --
        if i + 1 < n and plsql[i:i+2] == '--':
            # 跳过到行尾
            end_idx = plsql.find('\n', i)
            if end_idx != -1:
                i = end_idx  # 保留换行符
            else:
                i = n
            continue
        
        # 3. 检查 Dollar-quoted 字符串 (PL/pgSQL): $$...$$ 或 $tag$...$tag$
        if plsql[i] == '$':
            # 尝试匹配 $tag$ 格式
            dollar_match = _match_dollar_quote(plsql, i)
            if dollar_match:
                tag, end_idx = dollar_match
                # 保留整个 dollar-quoted 字符串（包括内部的注释）
                result.append(plsql[i:end_idx])
                i = end_idx
                continue
        
        # 4. 检查 Q-quoted 字符串 (Oracle): q'[...]', q'{...}', q'<...>', q'(...)'
        if i + 1 < n and plsql[i:i+2].lower() == "q'":
            q_end = _find_q_quote_end(plsql, i)
            if q_end != -1:
                result.append(plsql[i:q_end])
                i = q_end
                continue
        
        # 5. 检查 E-string (Postgres): E'...' 或 e'...'
        if i + 2 <= n and plsql[i:i+2].lower() == "e'":
            end_idx = _find_c_style_string_end(plsql, i)
            result.append(plsql[i:end_idx])
            i = end_idx
            continue

        # 6. 检查普通单引号字符串
        if plsql[i] == "'":
            end_idx = _find_string_end(plsql, i, "'")
            result.append(plsql[i:end_idx])
            i = end_idx
            continue
        
        # 7. 检查双引号标识符
        if plsql[i] == '"':
            end_idx = _find_string_end(plsql, i, '"')
            result.append(plsql[i:end_idx])
            i = end_idx
            continue
        
        # 8. 普通字符
        result.append(plsql[i])
        i += 1
    
    # 后处理：去除行尾空格和尾部空行
    processed = ''.join(result)
    lines = processed.split('\n')
    lines = [line.rstrip() for line in lines]
    
    # 移除尾部空行
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)


def _find_block_comment_end(s: str, start: int) -> int:
    '''
    找到块注释的结束位置，支持嵌套 (PostgreSQL 特性)。
    '''
    n = len(s)
    i = start + 2
    depth = 1
    
    while i < n:
        if i + 1 < n and s[i:i+2] == '/*':
            depth += 1
            i += 2
        elif i + 1 < n and s[i:i+2] == '*/':
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    
    return n


def _find_c_style_string_end(s: str, start: int) -> int:
    '''
    找到 PostgreSQL E-string (E'...') 的结束位置。
    支持反斜杠转义。
    '''
    n = len(s)
    i = start + 2  # Skip E'
    
    while i < n:
        if s[i] == '\\':
            # 安全地跳过转义字符
            if i + 1 < n:
                i += 2  # Skip escaped char
            else:
                i += 1  # 末尾的反斜杠，作为普通字符
        elif s[i] == "'":
            if i + 1 < n and s[i+1] == "'":
                i += 2  # Skip escaped single quote ''
            else:
                return i + 1  # End of string
        else:
            i += 1
    
    return n


def _match_dollar_quote(s: str, start: int) -> tuple:
    '''
    匹配 Dollar-quoted 字符串的开始标签，并找到对应的结束位置。
    支持 $$...$$ 和 $tag$...$tag$ 格式。
    
    Returns:
        (tag, end_idx) 或 None
    '''
    n = len(s)
    # 找到开始标签的结束位置
    j = start + 1
    while j < n and (s[j].isalnum() or s[j] == '_'):
        j += 1
    
    if j < n and s[j] == '$':
        tag = s[start:j+1]  # 包括两边的 $
        # 找到结束标签
        end_tag_pos = s.find(tag, j + 1)
        if end_tag_pos != -1:
            return (tag, end_tag_pos + len(tag))
    
    return None


def _find_q_quote_end(s: str, start: int) -> int:
    '''
    找到 Oracle Q-quoted 字符串的结束位置。
    格式: q'X...X' 其中 X 是配对的分隔符
    支持: [], {}, <>, (), 或任意字符
    '''
    if start + 3 >= len(s):
        return -1
    
    open_char = s[start + 2]
    # 确定配对的闭合字符
    pairs = {'[': ']', '{': '}', '<': '>', '(': ')'}
    close_char = pairs.get(open_char, open_char)
    
    # 查找 close_char + ' 的位置
    search_pattern = close_char + "'"
    end_idx = s.find(search_pattern, start + 3)
    
    if end_idx != -1:
        return end_idx + 2
    return -1


def _find_string_end(s: str, start: int, quote_char: str) -> int:
    '''
    找到字符串的结束位置，处理转义引号。
    '''
    n = len(s)
    i = start + 1
    
    while i < n:
        if s[i] == quote_char:
            # 检查是否是转义的引号
            if i + 1 < n and s[i + 1] == quote_char:
                i += 2  # 跳过转义引号
            else:
                return i + 1  # 字符串结束
        else:
            i += 1
    
    return n  # 未闭合，返回末尾