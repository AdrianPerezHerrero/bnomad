"""Check authored Markdown/notebook links, heading anchors and release metadata.

This offline check works in a checkout or unpacked source archive. External
references are inventoried for separate verification, never guessed from HTTP
errors or used to make the deterministic CI gate depend on a remote service.
"""
import argparse
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def markdown_text(path):
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix == '.ipynb':
        notebook = json.loads(text)
        text = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                         if cell['cell_type'] == 'markdown')
    # Fenced examples may deliberately contain placeholder links and headings.
    return re.sub(r'(?ms)^\s*(```|~~~).*?^\s*\1\s*$', '', text)


def link_targets(text):
    targets = []
    for match in re.finditer(r'!?\[[^\]\n]*\]\(', text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start < len(text) and text[start] == '<':
            stop = text.find('>', start+1)
            if stop >= 0:
                targets.append(text[start+1:stop])
            continue
        stop, depth = start, 0
        while stop < len(text):
            char = text[stop]
            if char == '\\':
                stop += 2
                continue
            if depth == 0 and (char == ')' or char.isspace()):
                break
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            stop += 1
        targets.append(text[start:stop])
    for match in re.finditer(r'(?m)^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))', text):
        targets.append(match[1] or match[2])
    targets.extend(re.findall(r'(?:href|src)=["\']([^"\']+)["\']', text))
    targets.extend(re.findall(r'<(https?://[^<>\s]+)>', text))
    # Plain URLs are clickable in GitHub prose too (for example attribution).
    targets.extend(match.rstrip('.,;') for match in re.findall(
        r'(?<![\w"\'(\]])https?://[^\s<>`]+', text))
    return targets


def heading_anchors(text):
    anchors = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    seen = {}
    for match in re.finditer(r'(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*$', text):
        heading = re.sub(r'!?\[([^\]]+)\]\([^)]*\)', r'\1', match[1])
        heading = re.sub(r'<[^>]*>', '', heading).lower()
        heading = ''.join(char for char in heading if char in '-_ ' or
                          unicodedata.category(char)[0] in ('L', 'N', 'M'))
        slug = heading.replace(' ', '-')
        duplicate = seen.get(slug, 0)
        seen[slug] = duplicate+1
        anchors.add(slug if duplicate == 0 else f'{slug}-{duplicate}')
    return anchors


def authored_files(root):
    paths = [root/name for name in ('README.md', 'CHANGELOG.md', 'NOTICE.md', 'GITHUB_SETUP.md')]
    for directory in ('docs', 'tutorials', 'examples/configurations'):
        for suffix in ('*.md', '*.ipynb'):
            paths.extend(path for path in (root/directory).rglob(suffix)
                         if not any(part.startswith('.') or part in ('baseline-notebooks', 'final-notebooks')
                                    for part in path.relative_to(root).parts))
    return sorted(set(path for path in paths if path.is_file()))


def audit(root):
    files = authored_files(root)
    issues, external, cache = [], {}, {}
    count = 0
    for path in files:
        text = markdown_text(path)
        for target in link_targets(text):
            count += 1
            url = urlsplit(target)
            if url.scheme or target.startswith('//'):
                if url.scheme in ('http', 'https'):
                    external.setdefault(target, set()).add(str(path.relative_to(root)))
                continue
            destination = path if not url.path else (path.parent/unquote(url.path)).resolve()
            if not destination.exists():
                issues.append(dict(file=str(path.relative_to(root)), target=target, error='missing path'))
                continue
            if url.fragment and destination.suffix in ('.md', '.ipynb'):
                if destination not in cache:
                    cache[destination] = heading_anchors(markdown_text(destination))
                if unquote(url.fragment) not in cache[destination]:
                    issues.append(dict(file=str(path.relative_to(root)), target=target, error='missing heading'))
    project = (root/'pyproject.toml').read_text(encoding='utf-8')
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', project)[1]
    citation = (root/'CITATION.cff').read_text(encoding='utf-8-sig')
    cited_version = re.search(r'(?m)^version:\s*["\']?([^\s"\']+)', citation)[1]
    if cited_version != version or f'**Version:** {version}' not in (root/'README.md').read_text(encoding='utf-8-sig'):
        issues.append(dict(file='release metadata', error='README/CITATION version differs from pyproject.toml'))
    retired_url = 'https://github.com/AdrianPerezHerrero/HDP-GPC'
    for name, text in [('pyproject.toml', project), ('CITATION.cff', citation)]:
        if retired_url in text:
            issues.append(dict(file=name, error='retired repository URL'))
    return dict(version=version, files=len(files), links=count, issues=issues,
                external_urls={url:sorted(paths) for url, paths in sorted(external.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Documentation {report['version']}: {report['files']} authored files, "
          f"{report['links']} links, {len(report['issues'])} issues")
    for issue in report['issues']:
        print(f"{issue['file']}: {issue['error']}: {issue.get('target', '')}")
    raise SystemExit(bool(report['issues']))


if __name__ == '__main__':
    main()
